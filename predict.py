# Standard library imports
import os
import json
import math
import random
import re
import pickle
import warnings
from datetime import datetime
from collections import defaultdict, Counter
from itertools import permutations
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

# External RNA tools
import RNA  # ViennaRNA

# Local module imports
import lib.tree_decomp as td

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置打印选项，确保打印所有张量元素
torch.set_printoptions(threshold=float("inf"))
torch.set_printoptions(sci_mode=False, precision=4)

# S\I\B\M\H\U
STRUCT_TYPE_TO_IDX = {
    'S': 0,  # stem
    'H': 1,  # hairpin
    'B': 2,  # bulge
    'I': 3,  # internal loop
    'M': 4,  # multi-loop
    'E': 5,  # external loop
    'X': 6  # unclassified / unpaired
}

NUM_STRUCT_TYPES = len(STRUCT_TYPE_TO_IDX)  # 7


# ==========================================
# 1. RNA 处理
# ==========================================
def process_rna(rna_name, rna_seq, rna_struct):
    """
    将 RNA 结构分解为接合树（Junction Tree）并提取节点和边信息。
    """
    # 清洗序列：去除多余符号、转大写、将 T 替换为 U
    processed_seq = rna_seq.replace('&', '').strip().upper().replace('T', 'U')
    processed_struct = rna_struct.replace('&', '').strip()

    # 使用 lib.tree_decomp (td) 进行分解
    # 注意：node_labels 通常是一个 list ['S', 'I', 'M'...]
    try:
        adjmat, node_labels, hpn_nodes_assignment = td.decompose(processed_struct)
    except Exception as e:
        print(f"Error in decomposition for {rna_name}: {e}")
        # 返回空结构以防止程序崩溃
        return [], np.array([[], []]), {}, {}, {}

    # 创建 RNA 接合树对象用于获取边遍历顺序
    tree = td.RNAJunctionTree(processed_seq, processed_struct)

    # 初始化栈用于 DFS 遍历，并从根节点（索引 0）开始
    stack = []
    if tree.nodes:
        td.dfs(stack, tree.nodes[0], 0)

    # 提取节点信息
    node_data = {}
    node_label = {}
    node_bases = {}

    # ================= 修正点开始 =================
    # node_labels 是 list，使用 enumerate 遍历
    for idx, label in enumerate(node_labels):
        # 兼容 hpn_nodes_assignment 可能是 dict 或 list 的情况
        if isinstance(hpn_nodes_assignment, dict):
            bases = hpn_nodes_assignment.get(idx, [])
        elif isinstance(hpn_nodes_assignment, list):
            bases = hpn_nodes_assignment[idx] if idx < len(hpn_nodes_assignment) else []
        else:
            bases = []

        node_label[idx] = label
        node_bases[idx] = bases
        node_data[idx] = (label, bases)
    # ================= 修正点结束 =================

    # 提取边信息（正向边与反向边）
    edges = []
    back_edges = []
    for edge_tuple in stack:
        node1, node2, direction = edge_tuple
        if direction == 1:  # 正向边
            edges.append((node1.idx, node2.idx))
        else:               # 反向边
            back_edges.append((node1.idx, node2.idx))

    # 转换为 numpy 数组格式的 edge_index
    edge_index = np.array(list(zip(*edges))) if edges else np.array([[], []])

    return stack, edge_index, node_label, node_bases, node_data


# ==========================================
# 2. 数据读取函数
# ==========================================
def read_rna_data(file_path):
    """
    从文本文件中读取 RNA 数据，支持多行读取及多种二级结构符号。

    Args:
        file_path (str): RNA 数据文件的路径。

    Returns:
        list: 包含 (rna_name, rna_seq, rna_struct) 元组的列表。
    """

    def is_sequence_line(line):
        """检查行是否只包含碱基字符"""
        return len(line) > 0 and all(c in 'AUGCaugct& ' for c in line)

    def is_structure_line(line):
        """检查行是否只包含二级结构符号"""
        return len(line) > 0 and all(c in '().[]{}<>|& ._:' for c in line)

    rna_data = []
    current_name, current_seq, current_struct = None, "", ""

    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # 如果当前已有读取好的数据，先保存
                if current_name is not None and current_seq and current_struct:
                    rna_data.append((current_name, current_seq, current_struct))

                # 重置变量读取下一个 RNA
                current_name = line[1:].strip()
                current_seq, current_struct = "", ""

            elif is_sequence_line(line):
                current_seq += line.replace(' ', '')

            elif is_structure_line(line):
                current_struct += line.replace(' ', '')

    # 别忘了添加最后一个 RNA
    if current_name is not None and current_seq and current_struct:
        rna_data.append((current_name, current_seq, current_struct))

    print(f"Successfully loaded {len(rna_data)} RNA records from {file_path}")
    return rna_data


# ==========================================
# 3. 辅助定位函数
# ==========================================
def find_pdb_file(rna_name, pdb_dir):
    """
    根据 RNA 名称在指定目录下查找对应的 PDB 结构文件。

    Args:
        rna_name (str): RNA 名称。
        pdb_dir (str): PDB 文件所在的根目录。

    Returns:
        str or None: 如果找到则返回完整路径，否则返回 None。
    """
    # 提取核心标识符（通常是空格前的第一部分）
    rfam_id = rna_name.split()[0]
    pdb_filename = f"{rfam_id}.pdb"
    pdb_path = os.path.join(pdb_dir, pdb_filename)

    if os.path.exists(pdb_path):
        return pdb_path
    return None


# ==========================================
# 4. PDB 坐标提取 (C4' 原子)
# ==========================================
def extract_bases_coordinates(pdb_file):
    """
    从 PDB 文件中提取所有残基 C4' 原子的坐标。

    Args:
        pdb_file (str): PDB 文件路径。

    Returns:
        dict: 键为残基序号(从0开始)，值为坐标元组 (x, y, z)。
    """
    base_coord = {}
    residue_counter = 0

    if not os.path.exists(pdb_file):
        return base_coord

    with open(pdb_file, 'r') as file:
        for line in file:
            # PDB 标准格式中，ATOM/HETATM 位于前6列，原子名在12-16列
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "C4'":
                    try:
                        # 按照 PDB 标准固定列宽提取坐标，比正则更安全
                        # x: 30-38, y: 38-46, z: 46-54
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        base_coord[residue_counter] = (x, y, z)
                        residue_counter += 1
                    except ValueError:
                        print(f"Warning: Failed to parse coordinates at line: {line.strip()}")

    return base_coord


# ==========================================
# 5. 节点坐标计算 (针对 Junction 与常规节点)
# ==========================================
def is_junction_node(bases):
    """
    判断是否为 Junction 节点（包含三个或以上子螺旋列表的节点）。
    """
    if isinstance(bases, list) and len(bases) > 0:
        # 如果第一个元素是列表，说明是“列表的列表”结构
        if isinstance(bases[0], list):
            return len(bases) >= 3
    return False


def get_junction_endpoints(bases):
    """
    获取 Junction 节点中每个子列表的首尾碱基索引。
    """
    endpoints = []
    for base_list in bases:
        if isinstance(base_list, list) and len(base_list) > 0:
            endpoints.extend([base_list[0], base_list[-1]])
    return endpoints


def calculate_node_coordinates(base_coord, node_bases):
    """
    计算接合树中每个节点的中心坐标。

    Args:
        base_coord (dict): 全局残基坐标字典。
        node_bases (dict): 节点对应的碱基索引字典。

    Returns:
        dict: 键为节点ID，值为计算出的平均坐标。
    """
    node_coord = {}

    for node, bases in node_bases.items():
        all_coords = []

        # 逻辑分支 1: Junction 节点 (采用端点平均法)
        #
        if is_junction_node(bases):
            endpoint_bases = get_junction_endpoints(bases)
            for b_idx in endpoint_bases:
                if b_idx in base_coord:
                    all_coords.append(base_coord[b_idx])

        # 逻辑分支 2: 非 Junction 节点 (采用所有成员平均法)
        else:
            if isinstance(bases, list):
                # 扁平化处理：无论嵌套与否，提取所有碱基
                for item in bases:
                    if isinstance(item, list):
                        for b_idx in item:
                            if b_idx in base_coord:
                                all_coords.append(base_coord[b_idx])
                    else:
                        if item in base_coord:
                            all_coords.append(base_coord[item])

        # 计算质心并保留三位小数
        if all_coords:
            mean_coords = np.mean(np.array(all_coords), axis=0)
            node_coord[node] = tuple(np.round(mean_coords, 3).tolist())

    return node_coord


# ==========================================
# 6. 3D 树状图解析与角度计算
# ==========================================
def read_3DTreeGraphpdb(filename):
    """
    读取 3DTreeGraph 的 PDB 文件，提取节点索引与对应的空间坐标。

    Args:
        filename (str): PDB 文件路径。

    Returns:
        dict: 键为节点索引（int），值为坐标元组 (x, y, z)。
    """
    coordinates = {}
    if not os.path.exists(filename):
        return coordinates

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                try:
                    # PDB 标准列宽解析
                    idx = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coordinates[idx] = (x, y, z)
                except ValueError:
                    continue
    return coordinates


def calculate_angle(p1, p2, p3):
    """
    计算由三个点 p1, p2, p3 构成的夹角（以 p2 为顶点）。

    Args:
        p1, p2, p3: 坐标元组或数组。

    Returns:
        float: 夹角角度值（0-180度）。
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0

    cosine_angle = np.dot(v1, v2) / norm_product
    # 数值稳定性：防止 cosine_angle 略微超过 [-1, 1] 范围导致 arccos 报错
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return float(np.degrees(angle))


def sort_angles_for_node(node_angles_list, angles_data):
    """
    为 Junction 节点按照定义的节点序（通常为逆时针或索引序）对角度进行一致性排序。
    这是确保 GNN 预测目标（alpha, beta, gamma）顺序一致的关键步骤。

    Args:
        node_angles_list (list): 当前中心节点关联的角度数值列表。
        angles_data (list): 完整的角度元组列表 [(n1, center, n3, angle_val), ...]。

    Returns:
        list: 排序后的角度数值列表。
    """
    if len(node_angles_list) <= 1:
        return node_angles_list

    # 提取与当前中心节点相关的完整信息
    relevant_angles = [a for a in angles_data if a[3] in node_angles_list]
    if not relevant_angles:
        return node_angles_list

    # 获取所有邻居节点并确定最小/最大节点编号作为参考
    connected_nodes = set()
    for angle in relevant_angles:
        connected_nodes.add(angle[0])
        connected_nodes.add(angle[2])

    min_node = min(connected_nodes)
    max_node = max(connected_nodes)

    sorted_angles = []

    # 排序规则（基于三路接头节点索引的约定）：
    # 1. 首先添加包含 min_node 但不包含 max_node 的角度
    for angle in relevant_angles:
        if (angle[0] == min_node or angle[2] == min_node) and \
                (angle[0] != max_node and angle[2] != max_node):
            sorted_angles.append(angle[3])

    # 2. 添加不包含 min_node 的角度（中间角度）
    for angle in relevant_angles:
        if angle[0] != min_node and angle[2] != min_node:
            sorted_angles.append(angle[3])

    # 3. 最后添加同时包含 min_node 和 max_node 的角度（跨越最小最大的闭合角）
    for angle in relevant_angles:
        if (angle[0] == min_node or angle[2] == min_node) and \
                (angle[0] == max_node or angle[2] == max_node):
            sorted_angles.append(angle[3])

    return sorted_angles


def process_node_angles(angles_raw_data):
    """
    对所有节点的角度进行聚合与规范化排序。

    Args:
        angles_raw_data (list): 原始计算出的角度元组列表。

    Returns:
        dict: 键为节点索引（0-based），值为排序后的角度列表。
    """
    node_angles_map = {}

    # 按照中心节点进行分组
    for entry in angles_raw_data:
        center_node = entry[1]
        if center_node not in node_angles_map:
            node_angles_map[center_node] = []
        node_angles_map[center_node].append(entry[3])

    final_sorted_map = {}
    for node_idx, val_list in node_angles_map.items():
        # 调用排序函数确保输出顺序为 [α, β, γ]
        sorted_vals = sort_angles_for_node(val_list, angles_raw_data)
        # 将节点索引转为 0-based 以匹配 GNN 图结构
        final_sorted_map[node_idx - 1] = sorted_vals

    return final_sorted_map


# ==========================================
# 7. Junction 角度计算
# ==========================================
def node_angle(pdb_filename, rna_name, stack, node_data, node_bases, new_mapping, base_coord, new_edge_index):
    """
    计算 Junction 节点的几何角度，并根据几何有效性修正节点类型。
    """
    original_coordinates = read_3DTreeGraphpdb(pdb_filename)
    # 调整为 0-based 索引以匹配内部逻辑
    coordinates = {k - 1: v for k, v in original_coordinates.items()}

    def has_enough_bases_for_angle(bases_lists):
        """检查是否有足够的碱基对（至少6个）来计算3个夹角"""
        count = sum(2 for bl in bases_lists if isinstance(bl, list) and len(bl) >= 2)
        return count >= 6

    def calculate_center_coord(start_idx, end_idx):
        """计算两个碱基坐标的中点"""
        p1, p2 = base_coord[start_idx], base_coord[end_idx]
        return tuple((np.array(p1) + np.array(p2)) / 2)

    def is_valid_junction_angles(angles):
        """验证三个角度之和是否接近 360 度（允许 1 度误差）"""
        return abs(sum(angles) - 360) <= 1

    # 创建反向映射与连接数统计
    reverse_mapping = {v: k for k, v in new_mapping.items()}

    def count_connections(node, edge_idx):
        return np.sum((edge_idx[0] == node) | (edge_idx[1] == node))

    # 识别 Junction 节点：必须有 3 个连接且碱基充足
    junction_nodes = {
        reverse_mapping[n]: b for n, b in node_bases.items()
        if isinstance(b[0], list) and len(b) >= 3 and
           has_enough_bases_for_angle(b) and count_connections(n, new_edge_index) == 3
    }

    # 节点类型修正逻辑
    updated_node_data = {}
    for node, (n_type, seqs) in node_data.items():
        conn = count_connections(node, new_edge_index)
        # 如果是多环（M）但连接数不为3，降级为内部环（I）
        if n_type == 'M' and conn != 3:
            updated_node_data[node] = ('I', seqs)
        else:
            updated_node_data[node] = (n_type, seqs)

    # 进一步区分凸环（B）与内部环（I）
    for node, (n_type, seqs) in list(updated_node_data.items()):
        if node != 0 and n_type == 'I':
            if any(len(s) == 2 for s in seqs):
                updated_node_data[node] = ('B', seqs)

    # 计算 Junction 角度
    junction_angles = {}
    for node, base_lists in junction_nodes.items():
        j_coord = coordinates[node]
        # 提取并排序碱基对
        pairs = sorted([b for bl in base_lists if isinstance(bl, list) for b in [bl[0], bl[-1]]])

        # 计算三个螺旋相对于中心点的中轴线向量中点
        centers = [
            calculate_center_coord(pairs[0], pairs[-1]),
            calculate_center_coord(pairs[1], pairs[2]),
            calculate_center_coord(pairs[3], pairs[4])
        ]

        angles_info = []
        for i in range(3):
            p_curr, p_next = centers[i], centers[(i + 1) % 3]
            deg = calculate_angle(p_curr, j_coord, p_next)
            angles_info.append((p_curr, j_coord, p_next, deg))

        if node in new_mapping:
            junction_angles[new_mapping[node]] = angles_info

    # 最终结果格式化与有效性检查
    all_angles_list = []
    final_node_angles = {}
    for node, ang_list in junction_angles.items():
        deg_values = [a[3] for a in ang_list]

        if not is_valid_junction_angles(deg_values):
            # 如果几何上不闭合，将该节点类型修正为 I，角度设为占位符 360
            if node in updated_node_data and updated_node_data[node][0] == 'M':
                updated_node_data[node] = ('I', updated_node_data[node][1])
                final_node_angles[node] = [360]
                continue

        final_node_angles[node] = deg_values
        all_angles_list.extend(ang_list)

    # 非 Junction 节点填充 360 度占位符
    for n in set(new_mapping.values()):
        if n not in final_node_angles:
            final_node_angles[n] = [360]

    return all_angles_list, final_node_angles, updated_node_data


# ==========================================
# 8. RNA 图转换（Stem 节点为边特征）
# ==========================================
def transform_rna_graph(rna_name, node_label, edge_index, node_data, node_bases):
    """
    将 RNA 原始图转换为简化图：将螺旋（Stem）节点塌陷为连接非螺旋节点之间的边。
    """
    stem_nodes = {i for i, lbl in node_label.items() if lbl == 'S'}
    non_stem_nodes = {i for i, lbl in node_label.items() if lbl != 'S'}

    # 建立新旧索引映射
    new_mapping = {old: i for i, old in enumerate(sorted(non_stem_nodes))}

    new_node_data = {new_mapping[o]: node_data[o] for o in non_stem_nodes}
    new_node_bases = {new_mapping[o]: node_bases[o] for o in non_stem_nodes}

    edges = list(zip(edge_index[0], edge_index[1]))
    stem_edges_found = []
    edge_features = {}
    direct_edges = []

    # 处理螺旋节点：将其转化为边信息
    #
    for s_node in stem_nodes:
        conn_edges = [e for e in edges if s_node in e]

        # 简化逻辑：寻找螺旋两端的非螺旋节点并连边
        neighbor_nodes = []
        for e in conn_edges:
            neighbor = e[0] if e[0] != s_node else e[1]
            neighbor_nodes.append(neighbor)

        # 如果螺旋连接了两个非螺旋节点
        for i in range(len(neighbor_nodes)):
            for j in range(i + 1, len(neighbor_nodes)):
                n1, n2 = neighbor_nodes[i], neighbor_nodes[j]
                if n1 not in stem_nodes and n2 not in stem_nodes:
                    new_e = tuple(sorted((n1, n2)))
                    if new_e not in stem_edges_found:
                        stem_edges_found.append(new_e)
                        edge_features[new_e] = {'sequence': node_data[s_node], 'bases': node_bases[s_node]}

    # 保留非螺旋节点之间的直接连接
    for u, v in edges:
        if u in non_stem_nodes and v in non_stem_nodes:
            direct_edges.append(tuple(sorted((u, v))))

    all_combined_edges = list(set(direct_edges + stem_edges_found))

    # 转换为新索引格式
    transformed_edges = [(new_mapping[u], new_mapping[v]) for u, v in all_combined_edges]
    new_edge_idx = np.array(list(zip(*transformed_edges))) if transformed_edges else np.array([[], []])

    # 更新特征字典的索引
    final_edge_features = {}
    for (u, v), feat in edge_features.items():
        final_edge_features[(new_mapping[u], new_mapping[v])] = feat

    return new_edge_idx, new_mapping, stem_edges_found, final_edge_features, new_node_data, new_node_bases


# ==========================================
# 9. 节点标签验证与修正 (基于预测几何)
# ==========================================
def validate_and_correct_node_labels(new_node_data, y_labels_tensor):
    """
    根据几何计算出的角度张量，验证并修正节点类型标签（主要针对 Multi-loop 节点）。

    Args:
        new_node_data (dict): 转换后的节点数据 {idx: (label, sequences)}。
        y_labels_tensor (torch.Tensor): 形状为 [N, 3] 的角度张量。

    Returns:
        dict: 修正后的节点数据字典。
    """
    corrected_node_data = new_node_data.copy()
    num_nodes = y_labels_tensor.size(0)

    for new_idx in range(num_nodes):
        if new_idx not in new_node_data:
            continue

        label, parts = new_node_data[new_idx]
        angles = y_labels_tensor[new_idx].tolist()  # [α, β, γ]

        # 统计非零角度的数量（容差 1e-5）
        nonzero_angles = [a for a in angles if abs(a) > 1e-5]
        num_nonzero = len(nonzero_angles)

        # 核心逻辑：只有拥有 3 个有效夹角的节点才允许保留 'M' (Multi-loop) 标签
        if label == 'M':
            if num_nonzero != 3:
                # 识别为误判：如果只有 2 个部分，降级为 'I' (Internal loop)
                if len(parts) == 2:
                    corrected_node_data[new_idx] = ('I', parts)
                else:
                    # 如果几何上无法形成闭合三路接头，打印警告并根据模型稳健性决定是否降级
                    print(f"Warning: Node {new_idx} ('M') has {num_nonzero} non-zero angles. Geometry is invalid.")
                    corrected_node_data[new_idx] = ('I', parts)

        # 补充：对 I 或 P 类型节点的几何一致性提示
        elif label in ['I', 'P'] and num_nonzero > 2:
            pass  # 可以在此添加调试信息

    return corrected_node_data


# ==========================================
# 10. 核苷酸碱基图构建
# ==========================================
def parse_base_pairs_with_type(secstruct):
    """
    解析二级结构字符串（支持假结符号 [] {} <> 等）。
    """
    bps = []
    is_pk = []
    # 括号匹配映射
    pairs_map = {'(': ')', '[': ']', '{': '}', '<': '>'}
    stacks = {k: [] for k in pairs_map}

    for i, char in enumerate(secstruct):
        if char in pairs_map:
            stacks[char].append(i)
        elif char in pairs_map.values():
            for open_char, close_char in pairs_map.items():
                if char == close_char:
                    if stacks[open_char]:
                        j = stacks[open_char].pop()
                        bps.append((j, i))
                        is_pk.append(open_char != '(')  # '(' 之外的均视为假结
                    break
    return bps, is_pk


def get_rna_edge_index(rna_seq, rna_struct):
    """
    构建 PyG 图格式的核苷酸图。

    特征维度定义：
    - Node Attr [N, 5]: One-hot (A, U, G, C, N)
    - Edge Attr [E, 4]: One-hot (Watson-Crick, Wobble/Non-canonical, Pseudoknot, Backbone)
    """
    # 预处理：清洗并规范化输入
    clean_seq = re.sub(r'[^AUCGN]', 'N', rna_seq.upper())
    clean_struct = rna_struct  # 假设已清洗

    L = len(clean_seq)
    if L == 0:
        return torch.empty(2, 0), torch.empty(0, 4), torch.empty(0, 5)

    # 1. 节点特征构建
    node_map = {'A': [1, 0, 0, 0, 0], 'U': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0]}
    node_attr = torch.tensor([node_map.get(nt, [0, 0, 0, 0, 1]) for nt in clean_seq], dtype=torch.float)

    # 2. 边与特征构建
    edge_list, edge_attrs = [], []

    def add_bidirectional_edge(i, j, attr):
        edge_list.extend([(i, j), (j, i)])
        edge_attrs.extend([attr, attr])

    # A. 骨架边 (Backbone)
    for i in range(L - 1):
        add_bidirectional_edge(i, i + 1, [0.0, 0.0, 0.0, 1.0])

    # B. 碱基配对边 (Base Pairs)
    bps, pk_flags = parse_base_pairs_with_type(clean_struct)
    canonical = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}

    for (i, j), is_pk in zip(bps, pk_flags):
        if is_pk:
            add_bidirectional_edge(i, j, [0.0, 0.0, 1.0, 0.0])  # 假结边
        else:
            pair = (clean_seq[i], clean_seq[j])
            if pair in canonical:
                add_bidirectional_edge(i, j, [1.0, 0.0, 0.0, 0.0])  # 标准配对
            else:
                add_bidirectional_edge(i, j, [0.0, 1.0, 0.0, 0.0])  # 非标准配对

    # 转换张量
    if not edge_list:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, 4)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return edge_index, edge_attr, node_attr


# ==========================================
# 11. 辅助函数：碱基图结构标签构建
# ==========================================
def encode_struct_onehot(micro_struct_str):
    """将碱基结构字符串序列转换为 One-hot 张量"""
    L = len(micro_struct_str)
    onehot = torch.zeros(L, NUM_STRUCT_TYPES, dtype=torch.float)
    for i, typ in enumerate(micro_struct_str):
        idx = STRUCT_TYPE_TO_IDX.get(typ, STRUCT_TYPE_TO_IDX['X'])
        onehot[i, idx] = 1.0
    return onehot  # (L, NUM_STRUCT_TYPES)


def build_micro_struct_labels(node_data, seq_length):
    """
    根据树图节点信息构建全序列的碱基结构标签列表。
    例如：['S', 'S', 'I', 'I', 'S', 'S', ...]
    """
    micro_struct_str = ['X'] * seq_length

    for struct_id, (struct_type, positions) in node_data.items():
        all_positions = []
        if isinstance(positions, list):
            if not positions: continue
            # 展平嵌套列表 [[1,2], [3,4]] -> [1,2,3,4]
            if isinstance(positions[0], list):
                for sublist in positions:
                    all_positions.extend(sublist)
            else:
                all_positions = positions
        else:
            continue

        # 映射类型：P(Pseudoknot) 视为 S(Stem)
        label = 'S' if struct_type == 'P' else struct_type
        if label not in STRUCT_TYPE_TO_IDX: label = 'X'

        for pos in all_positions:
            if isinstance(pos, int) and 0 <= pos < seq_length:
                micro_struct_str[pos] = label

    return micro_struct_str


# ==========================================
# 12. RNA特征处理器
# ==========================================
class RNAFeatureProcessor:
    """
    负责提取 RNA 树图节点的生物学特征。
    """

    # ----------------------------------------
    # 基础与归一化逻辑
    # ----------------------------------------
    def one_hot_node_type(self, node_type, valid_types=('P', 'I', 'B', 'M', 'H')):
        """节点类型 One-hot 编码"""
        vec = [0] * len(valid_types)
        if node_type in valid_types:
            vec[valid_types.index(node_type)] = 1
        return vec

    def _get_basic_features(self, node_type, degree):
        """基础拓扑特征"""
        return self.one_hot_node_type(node_type)

    def get_node_dot_bracket(self, rna_struct, node_bases_index):
        """根据索引提取对应的点括号子串"""
        if not node_bases_index: return []
        if isinstance(node_bases_index[0], list):  # 嵌套列表
            return [[rna_struct[i] for i in seg] for seg in node_bases_index]
        else:
            return [[rna_struct[i] for i in node_bases_index]]

    def _normalize_data_to_triplet(self, data, node_type, degree, invalid_val='-1'):
        """
        核心逻辑：将任意类型的 Loop 数据（序列/结构/索引）标准化为三元组 [L1, L2, L3]。
        针对 Junction (M) 保留三个分支，其他类型进行填充或合并。
        """
        if not data: return [invalid_val] * 3

        # 针对不同节点类型和度数的标准化策略
        if node_type == 'M':
            if degree == 3:  # 标准三路接头
                if len(data) == 3:
                    return data
                elif len(data) == 4:
                    return [data[0] + data[3], data[1], data[2]]  # 合并首尾
                elif len(data) > 4:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]  # 合并剩余
            else:  # 非标准 M
                if len(data) <= 3:
                    return data + [invalid_val] * (3 - len(data))
                else:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]

        # 二路节点 (I, B) -> 填充第三项
        elif node_type in ['I', 'B']:
            if degree == 2:
                if len(data) == 2:
                    return [data[0], data[1], invalid_val]
                elif len(data) > 2:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]
            else:
                return data[:2] + [invalid_val] if len(data) >= 2 else data + [invalid_val] * (3 - len(data))

        # 单路节点 (H) -> 合并为一项，填充后两项
        elif node_type == 'H':
            return [sum(data, type(data[0])()), invalid_val, invalid_val]

        # 默认处理
        if len(data) == 1:
            return [data[0], invalid_val, invalid_val]
        elif len(data) == 2:
            return [data[0], data[1], invalid_val]
        return [data[0], data[1], sum(data[2:], type(data[0])())]

    def _clean_triplet(self, triplet):
        """移除 '&' 连接符并处理无效值"""
        cleaned = []
        for item in triplet:
            if isinstance(item, str):
                s = item.replace('&', '')
                cleaned.append(s if s else '-1')
            elif isinstance(item, list):  # 索引列表
                cleaned.append(item if item else [])
            else:
                cleaned.append(item)
        return cleaned

    # ----------------------------------------
    # 特征计算子函数
    # ----------------------------------------
    def _calc_length_feats(self, bases):
        """计算长度、不对称性、比例等几何约束特征"""
        lens = [len(b) if b != '-1' else 0 for b in bases]
        L1, L2, L3 = lens
        sorted_L = sorted(lens)
        total = sum(lens) + 1e-10

        return {
            'L1': L1, 'L2': L2, 'L3': L3,
            'min_L1_L2': min(L1, L2), 'min_L2_L3': min(L2, L3), 'min_L1_L3': min(L1, L3),
            'diff_L1_L2': abs(L1 - L2), 'diff_L2_L3': abs(L2 - L3), 'diff_L1_L3': abs(L1 - L3),
            'ratio_L1_L2': L1 / (L2 + 1e-10), 'ratio_L2_L3': L2 / (L3 + 1e-10), 'ratio_L1_L3': L1 / (L3 + 1e-10),
            'L1_prop': L1 / total, 'L2_prop': L2 / total, 'L3_prop': L3 / total,
            'L_min': sorted_L[0], 'L_mid': sorted_L[1], 'L_max': sorted_L[2],
            'L_span': sorted_L[2] - sorted_L[0]
        }

    def _calc_composition_feats(self, bases):
        """计算碱基组成 (A/G/C/U/Purine)"""
        feats = {}
        purines = {'A', 'G'}
        for i, branch in enumerate(['L1', 'L2', 'L3']):
            seq = bases[i]
            if seq == '-1' or not seq:
                for k in ['A', 'G', 'C', 'U', 'Purine', 'GC']: feats[f'{k}_{branch}'] = 0
                feats[f'max_consec_A_{branch}'] = 0
                continue

            tot = len(seq)
            counts = Counter(seq)
            feats[f'A_{branch}'] = counts['A'] / tot
            feats[f'G_{branch}'] = counts['G'] / tot
            feats[f'C_{branch}'] = counts['C'] / tot
            feats[f'U_{branch}'] = counts['U'] / tot
            feats[f'Purine_{branch}'] = (counts['A'] + counts['G']) / tot
            feats[f'GC_{branch}'] = (counts['G'] + counts['C']) / tot

            # 连续 A 模体检测 (A-minor motif indicator)
            matches = re.findall('A+', seq)
            feats[f'max_consec_A_{branch}'] = max((len(m) for m in matches), default=0)
        return feats

    def _calc_physicochemical_feats(self, bases):
        """计算物理化学特征：U-run (柔性), 电荷梯度"""
        feats = {}
        for i, branch in enumerate(['L1', 'L2', 'L3']):
            seq = bases[i]
            if seq == '-1' or len(seq) < 2:
                feats[f'U_run_{branch}'] = 0
                feats[f'Charge_grad_{branch}'] = 0
                continue

            # U-rich 区域检测
            u_runs = re.findall('U+', seq)
            feats[f'U_run_{branch}'] = max((len(r) for r in u_runs), default=0)

            # 电荷梯度
            eiips = [self.EIIP_dict.get(b, 0) for b in seq]
            feats[f'Charge_grad_{branch}'] = eiips[-1] - eiips[0]
        return feats

    def _encode_flanking(self, seq):
        """编码分支末端碱基 (Stacking 能量相关)"""
        mapping = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
        if not seq or seq == '-1': return 0, 0
        return mapping.get(seq[0], 0), mapping.get(seq[-1], 0)

    # ----------------------------------------
    # 主处理函数
    # ----------------------------------------
    def _get_junction_features(self, node_type, node_id, G, bases, node_bases_index, dot_bracket):
        """提取并组合所有高级特征"""
        # 1. 预处理：去除闭合对（第一个和最后一个碱基）
        # 这是为了只关注 Loop 内部的单链区域
        inner_bases = [b[1:-1] if len(b) > 2 else "" for b in bases]

        # 2. 标准化为三元组
        norm_bases = self._clean_triplet(
            self._normalize_data_to_triplet(inner_bases, node_type, G.degree(node_id), '-1')
        )

        # 3. 计算各类特征
        f_len = self._calc_length_feats(norm_bases)
        f_comp = self._calc_composition_feats(norm_bases)
        f_phys = self._calc_physicochemical_feats(norm_bases)

        # 4. Flanking base features
        flank_feats = []
        for i, seq in enumerate(norm_bases):
            s, e = self._encode_flanking(seq)
            flank_feats.extend([s, e])

        # 5. 组装特征向量 (保持固定顺序)
        # Lengths (12) + Composition (21) + Phys (6) + Flanking (6) + Sorted (4) ...
        # 这里只列出部分关键特征，需与你的模型输入维度对齐
        feature_vector = [
            f_len['L1'], f_len['L2'], f_len['L3'],
            f_len['min_L1_L2'], f_len['min_L2_L3'], f_len['min_L1_L3'],
            f_len['diff_L1_L2'], f_len['diff_L2_L3'], f_len['diff_L1_L3'],
            f_len['ratio_L1_L2'], f_len['ratio_L2_L3'], f_len['ratio_L1_L3'],
            f_len['L1_prop'], f_len['L2_prop'], f_len['L3_prop'],

            f_comp['Purine_L1'], f_comp['Purine_L2'], f_comp['Purine_L3'],
            f_comp['A_L1'], f_comp['A_L2'], f_comp['A_L3'],

            f_len['L_min'], f_len['L_mid'], f_len['L_max'], f_len['L_span'],

            f_comp['max_consec_A_L1'], f_comp['max_consec_A_L2'], f_comp['max_consec_A_L3'],

            # Flanking bases
            *flank_feats
        ]

        return feature_vector

    def process_structure(self, rna_name, rna_struct, node_data, node_type, bases, node_id, G, node_bases_index,
                          edge_data):
        """对外接口：计算指定节点的完整特征向量"""
        # 获取点括号表示
        dot_bracket = self.get_node_dot_bracket(rna_struct, node_bases_index[node_id])

        # 修正节点类型逻辑
        degree = G.degree(node_id)
        stem_conns = sum(1 for edge in edge_data.keys() if node_id in edge)

        effective_type = node_type
        # 如果是 M 但只有 2 个 Stem 连接，视作 I (Internal Loop)
        if node_type == 'M' and stem_conns == 2:
            effective_type = 'I'
        elif degree == 3 and len(bases) >= 4:
            effective_type = 'I'

        # 计算特征
        # basic_vec = self._get_basic_features(effective_type, degree)
        junc_vec = self._get_junction_features(effective_type, node_id, G, bases, node_bases_index[node_id],
                                               dot_bracket)

        return np.array(junc_vec, dtype=np.float32)


# ==========================================
# 13. 辅助计算
# ==========================================
def get_max_consecutive_length(indices):
    """计算索引列表中最长连续子序列的长度"""
    if not indices: return 0
    max_len = 1
    curr_len = 1
    sorted_indices = sorted(indices)
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i - 1] + 1:
            curr_len += 1
        else:
            max_len = max(max_len, curr_len)
            curr_len = 1
    return max(max_len, curr_len)


# ==========================================
# 14. 边特征计算 (Stem Features)
# ==========================================
def calculate_edge_features(rna_name, edge_data, rna_seq):
    """
    计算连接两个宏观节点的边（即 Stem）的特征。
    """
    features = {
        'stem_length': 0, 'gc_pairs': 0, 'au_pairs': 0, 'gu_pairs': 0,
        'gc_ratio': 0, 'au_ratio': 0, 'gu_ratio': 0,
        'hydrogen_bonds': 0,
        'consecutive_gc': 0, 'consecutive_au': 0, 'consecutive_gu': 0,
    }

    bases_data = edge_data.get('bases', [[], []])
    # 如果数据为空，返回 11 维零向量
    if not bases_data or not bases_data[0]:
        return [0.0] * 11 

    # 处理 Stem 数据格式
    if isinstance(bases_data[0], list):
        stem_length = len(bases_data[0])
        idx_list1 = bases_data[0]
        idx_list2 = bases_data[1]
    else: 
        stem_length = 1
        idx_list1 = [bases_data[0]]
        idx_list2 = [bases_data[1]]
    
    features['stem_length'] = stem_length

    # 提取序列并配对
    try:
        seq1 = [rna_seq[i].upper() for i in idx_list1]
        seq2 = [rna_seq[i].upper() for i in idx_list2]
        seq2 = seq2[::-1] 
    except IndexError:
        print(f"Warning: Index out of range in edge calculation for {rna_name}")
        return [0.0] * 11

    # 统计配对类型
    gc_indices, au_indices, gu_indices = [], [], []
    for i, (b1, b2) in enumerate(zip(seq1, seq2)):
        pair = tuple(sorted((b1, b2)))
        if pair == ('C', 'G'):
            features['gc_pairs'] += 1
            features['hydrogen_bonds'] += 3
            gc_indices.append(i)
        elif pair == ('A', 'U'):
            features['au_pairs'] += 1
            features['hydrogen_bonds'] += 2
            au_indices.append(i)
        elif pair == ('G', 'U'):
            features['gu_pairs'] += 1
            features['hydrogen_bonds'] += 1
            gu_indices.append(i)
    
    if stem_length > 0:
        features['gc_ratio'] = features['gc_pairs'] / stem_length
        features['au_ratio'] = features['au_pairs'] / stem_length
        features['gu_ratio'] = features['gu_pairs'] / stem_length
        features['consecutive_gc'] = get_max_consecutive_length(gc_indices)
        features['consecutive_au'] = get_max_consecutive_length(au_indices)
        features['consecutive_gu'] = get_max_consecutive_length(gu_indices)

    # === 返回特征 ===
    return [
        features['stem_length'],      # 1
        features['gc_pairs'],         # 2
        features['au_pairs'],         # 3
        features['gu_pairs'],         # 4
        features['gc_ratio'],         # 5
        features['au_ratio'],         # 6
        features['gu_ratio'],         # 7
        features['hydrogen_bonds'],   # 8
        features['consecutive_gc'],   # 9
        features['consecutive_au'],   # 10
        features['consecutive_gu'],   # 11
    ]


def process_all_edges(rna_name, edge_data_dict, edge_index, rna_seq):
    """批量处理所有边特征"""
    # 确保边顺序与 edge_index 一致
    # edge_index: [2, E] tensor
    num_edges = edge_index.size(1)
    edge_features = []

    # 构建查找表：(u, v) -> feature_data
    data_lookup = {}
    for (u, v), data in edge_data_dict.items():
        data_lookup[(u, v)] = data
        data_lookup[(v, u)] = data

    for i in range(num_edges):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if (u, v) in data_lookup:
            feat = calculate_edge_features(rna_name, data_lookup[(u, v)], rna_seq)
        else:
            feat = [0.0] * 12
        edge_features.append(feat)

    return torch.tensor(edge_features, dtype=torch.float), edge_index


# ==========================================
# 15. 节点特征计算 (Loop Features)
# ==========================================
def calculate_node_features(rna_name, rna_struct, node_data, node_bases_index, edge_index, edge_data, rna_seq):
    """
    计算所有节点的特征矩阵。
    """
    feature_processor = RNAFeatureProcessor()

    # 构建 NetworkX 图用于计算度数 (Degree)
    G = nx.Graph()
    G.add_nodes_from(node_data.keys())
    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        G.add_edges_from(zip(src, dst))
    else:  # 兼容 numpy
        G.add_edges_from(zip(edge_index[0], edge_index[1]))

    node_ids = sorted(node_data.keys())
    features_list = []

    # 辅助：获取序列片段
    def get_seqs(indices):
        if not indices: return []
        if isinstance(indices[0], list):  # 嵌套
            return [''.join([rna_seq[i] for i in seg]) for seg in indices]
        else:  # 扁平
            return [''.join([rna_seq[i] for i in indices])]

    for node_id in node_ids:
        try:
            node_type, _ = node_data[node_id]  # node_data: {id: (type, indices_raw)}
            indices = node_bases_index[node_id]
            bases_seqs = get_seqs(indices)

            # 调用 Processor
            # process_structure(self, rna_name, rna_struct, node_data, node_type, bases, node_id, G, node_bases_index, edge_data)
            feat_vec = feature_processor.process_structure(
                rna_name=rna_name,
                rna_struct=rna_struct,
                node_data=node_data,
                node_type=node_type,
                bases=bases_seqs,
                node_id=node_id,
                G=G,
                node_bases_index=node_bases_index,  # 传入完整字典
                edge_data=edge_data  # 传入完整边字典用于计算连接数
            )
            features_list.append(feat_vec)

        except Exception as e:
            print(f"Error processing node {node_id} in {rna_name}: {e}")
            features_list.append(np.zeros(34, dtype=np.float32))

    if not features_list:
        return torch.tensor([], dtype=torch.float)

    return torch.tensor(np.array(features_list), dtype=torch.float)


# ==========================================
# 16. 自定义图数据对象 (Data Wrapper)
# ==========================================
class RNADualGraphData(Data):
    """
    R3J-AGNN 的核心数据结构，包含核苷酸和树图两个层面的图信息。
    继承自 torch_geometric.data.Data 以支持自动 Batching。
    """

    def __init__(self,
                 # === 碱基图 (Atomic/Nucleotide Level) ===
                 micro_x=None,  # [L, 5]: 节点特征 (One-hot Base)
                 micro_struct_attr=None,  # [L, 7]: 结构类型特征 (One-hot Loop Type)
                 micro_edge_index=None,  # [2, E_micro]: 边索引
                 micro_edge_attr=None,  # [E_micro, 4]: 边特征 (Pair Type)

                 # === 树图 (Motif/Topology Level) ===
                 macro_x=None,  # [N_macro, F_node]: 节点特征
                 macro_edge_index=None,  # [2, E_macro]: 边索引
                 macro_edge_attr=None,  # [E_macro, F_edge]: 边特征

                 loop_label=None,  # [N_macro]: 节点类型标签 (用于 Loss Masking)

                 # === 跨层映射 (Mapping) ===
                 micro_to_macro=None,  # [L]: 每个核苷酸所属的树图节点或边索引

                 # === 元数据 (Metadata) ===
                 rna_name=None,
                 rna_seq=None,
                 rna_struct=None,
                 seq_length=None,
                 y=None,  # [N_macro, 3/6]: 预测目标 (角度)

                 **kwargs):
        super().__init__(**kwargs)

        # 赋值
        self.micro_x = micro_x
        self.micro_struct_attr = micro_struct_attr
        self.micro_edge_index = micro_edge_index
        self.micro_edge_attr = micro_edge_attr

        self.macro_x = macro_x
        self.macro_edge_index = macro_edge_index
        self.macro_edge_attr = macro_edge_attr
        self.loop_label = loop_label

        self.micro_to_macro = micro_to_macro

        self.rna_name = rna_name
        self.rna_seq = rna_seq
        self.rna_struct = rna_struct
        self.seq_length = seq_length
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        """
        PyG Batching 的关键逻辑：定义各属性在拼接时的增量。
        """
        if key == 'micro_edge_index':
            return self.seq_length

        if key == 'macro_edge_index':
            return self.macro_x.size(0) if self.macro_x is not None else 0

        if key == 'micro_to_macro':
            return self.macro_x.size(0) if self.macro_x is not None else 0

        return super().__inc__(key, value, *args, **kwargs)


# ==========================================
# 17. 构建映射 (Assignment Matrix)
# ==========================================
def create_micro_to_macro_mapping(
        new_node_data,  # {macro_idx: (type, bases_list)}
        original_node_bases,  # {old_tree_id: (type, bases_list)}
        new_mapping,  # {old_tree_id -> macro_idx}
        seq_length,
        original_edge_index  # 原始接合树的边索引
):
    """
    构建 micro_to_macro 张量，将每个核苷酸分配给树图的节点或边。
    - 正数 (>=0): 对应节点的索引
    - 负数 (<0):  对应边的索引 (用于 Stem)
    - -100:       未分配/无效
    """
    # 初始化映射张量
    mapping = torch.full((seq_length,), -100, dtype=torch.long)

    def flatten_indices(obj):
        """递归展平索引列表"""
        if isinstance(obj, int): return [obj]
        res = []
        for item in obj:
            res.extend(flatten_indices(item))
        return res

    # 1. 映射树图节点 (Loop/Junction)
    for macro_idx, (n_type, bases) in new_node_data.items():
        if not bases: continue
        for idx in flatten_indices(bases):
            if 0 <= idx < seq_length:
                mapping[idx] = macro_idx

    # 2. 构建原始树的邻接表 (用于查找 Stem 的邻居)
    adj = defaultdict(list)
    if isinstance(original_edge_index, np.ndarray):
        edge_iter = zip(original_edge_index[0], original_edge_index[1])
    else:
        edge_iter = []

    for u, v in edge_iter:
        adj[u].append(v)
        adj[v].append(u)

    # 3. 映射边 (Stem)
    stem_edge_map = {}  # debug 用
    stem_counter = 0

    for old_id, (n_type, bases) in original_node_bases.items():
        if old_id in new_mapping: continue  # 已处理的 Loop 节点

        # 这是一个 Stem 节点
        neighbors = [n for n in adj[old_id] if n in new_mapping]

        # 只有连接了两个有效 Loop 的 Stem 才被视为有效的边
        if len(neighbors) == 2:
            macro_u = new_mapping[neighbors[0]]
            macro_v = new_mapping[neighbors[1]]

            stem_key = -(stem_counter + 1)

            for idx in flatten_indices(bases):
                if 0 <= idx < seq_length and mapping[idx] == -100:  # 仅填充未占用位置
                    mapping[idx] = stem_key

            stem_edge_map[stem_key] = (macro_u, macro_v)
            stem_counter += 1

    return mapping, stem_edge_map


# 角度大小顺序
def angles_to_rank_labels(angles_tensor, mask=None):
    """
    将角度张量转换为排序标签（1=最小，2=中间，3=最大）
    对于全零行（无效角度），保留为 0。

    Args:
        angles_tensor: (N, 3) tensor of angles in degrees
        mask: (N,) bool tensor, True=有效节点。若为 None，则自动根据是否全零判断。

    Returns:
        rank_labels: (N, 3) long tensor, values in {0, 1, 2, 3}，其中 0 表示无效
    """
    N = angles_tensor.shape[0]
    device = angles_tensor.device

    # 默认：非全零行为有效
    if mask is None:
        # 检查每行是否全为 0
        mask = ~torch.all(angles_tensor == 0, dim=1)  # (N,)

    rank_labels = torch.zeros(N, 3, dtype=torch.long, device=device)

    valid_angles = angles_tensor[mask]  # (M, 3)
    M = valid_angles.shape[0]

    if M > 0:
        # 对每行的三个角度升序排序，得到索引
        sorted_indices = torch.argsort(valid_angles, dim=1)  # (M, 3)

        # 创建 [1,2,3] 的 rank 值
        rank_values = torch.tensor([1, 2, 3], device=device, dtype=torch.long).expand(M, -1)

        # 将 rank 值按 sorted_indices 的逆映射填回原位置
        # 方法：用 scatter 将 rank_values 分配到 sorted_indices 指定的位置
        ranks = torch.zeros_like(sorted_indices)
        ranks.scatter_(1, sorted_indices, rank_values)

        rank_labels[mask] = ranks

    return rank_labels


# ==========================================
# 18. 主推理管道 (prepare_data)
# ==========================================
def prepare_data(rna_name, rna_seq, rna_struct):
    """
    单样本推理数据准备流程：
    1. 解析序列与结构
    2. 生成碱基图 (Base Graph)
    3. 生成树图 (Motif Graph)
    4. 提取双层特征
    5. 封装为 PyG Data 对象
    """
    try:
        # 预处理：清洗序列
        clean_seq = re.sub(r'[^AUCGaucg]', '', rna_seq).upper()
        if len(clean_seq) != len(rna_struct):
            clean_seq = rna_seq.replace('&', '').upper()
            rna_struct = rna_struct.replace('&', '')

        seq_length = len(clean_seq)
        if seq_length == 0: return None

        # ------------------------------------
        # Step 1: 原始树分解 (Junction Tree Decomposition)
        # ------------------------------------
        stack, edge_index, node_label, node_bases, node_data = process_rna(rna_name, clean_seq, rna_struct)

        # ------------------------------------
        # Step 2: 树图图转换
        # ------------------------------------
        new_edge_index, new_mapping, stem_edges, edge_data, new_node_data, new_node_bases_index = \
            transform_rna_graph(rna_name, node_label, edge_index, node_data, node_bases)

        # ------------------------------------
        # Step 3: 碱基图构建 (Atomic Graph)
        # ------------------------------------
        # 获取碱基对和骨架连接
        secstruct_edge_indices, micro_edge_attr, micro_x = get_rna_edge_index(clean_seq, rna_struct)

        # 转换索引为 Tensor
        micro_edge_index = torch.tensor(secstruct_edge_indices, dtype=torch.long)

        # 构建节点的结构标签 (S, H, I, M...)
        micro_struct_str = build_micro_struct_labels(new_node_data, seq_length)
        micro_struct_attr = encode_struct_onehot(micro_struct_str)

        # ------------------------------------
        # Step 4: 树图特征提取
        # ------------------------------------
        # 4.1 边特征 (Stem 属性)
        edge_index_tensor = torch.tensor(new_edge_index, dtype=torch.long)
        macro_edge_attr, _ = process_all_edges(rna_name, edge_data, edge_index_tensor, clean_seq)

        # 4.2 节点特征 (Loop 属性)
        macro_x = calculate_node_features(
            rna_name, rna_struct, new_node_data, new_node_bases_index,
            new_edge_index, edge_data, clean_seq
        )

        # ------------------------------------
        # Step 5: 标签与映射
        # ------------------------------------
        # 生成 Loop Label (H=0, I=1, M=2...)
        node_type_map = {'H': 0, 'I': 1, 'M': 2, 'P': 3, 'S': 4, 'E': 5, 'B': 6, 'X': 7}
        sorted_node_ids = sorted(new_node_data.keys())
        loop_label_list = [node_type_map.get(new_node_data[nid][0], 7) for nid in sorted_node_ids]
        loop_label = torch.tensor(loop_label_list, dtype=torch.long)

        # 映射
        micro_to_macro, _ = create_micro_to_macro_mapping(
            new_node_data, node_data, new_mapping, seq_length, torch.tensor(edge_index)
        )

        # ------------------------------------
        # Step 6: 封装
        # ------------------------------------
        data = RNADualGraphData(
            # 碱基图
            micro_x=micro_x,
            micro_struct_attr=micro_struct_attr,
            micro_edge_index=micro_edge_index,
            micro_edge_attr=micro_edge_attr,

            # 树图
            macro_x=macro_x,
            macro_edge_index=edge_index_tensor,
            macro_edge_attr=macro_edge_attr,
            loop_label=loop_label,

            # 关联
            micro_to_macro=micro_to_macro,

            # 元数据
            rna_name=rna_name,
            rna_seq=rna_seq,
            rna_struct=rna_struct,
            seq_length=seq_length,

            # 预测模式下，标签为空
            y=None
        )

        return data

    except Exception as e:
        print(f"Error processing inference data {rna_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# 解析自定义 FASTA 文件
# ==========================================
def parse_rna_fasta(file_path):
    records = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return records

    with open(file_path, 'r') as f:
        # 过滤空行并去除首尾空格
        lines = [line.strip() for line in f if line.strip()]

    # Line 1: >Name
    # Line 2: Sequence
    # Line 3: Structure
    for i in range(0, len(lines), 3):
        try:
            header = lines[i]
            seq = lines[i + 1]
            struct = lines[i + 2]

            if header.startswith('>'):
                # 提取名字，去掉 '>' 和可能的参数
                name = header[1:].split()[0]
                records.append({
                    'name': name,
                    'seq': seq,
                    'struct': struct
                })
        except IndexError:
            print("Warning: File ended unexpectedly or format is inconsistent.")
            break

    print(f"Loaded {len(records)} RNA entries from file.")
    return records

