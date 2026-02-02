# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import torch
from torch_geometric.data import Batch

from predict import prepare_data
from model import DualGraphRNAModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="R3J-AGNN: Geometry-aware prediction of RNA three-way junctions.")

    # 输入输出相关
    parser.add_argument("--input", type=str, default="example/example.fasta",
                        help="Path to input FASTA file (Header, Seq, Struct).")
    parser.add_argument("--model", type=str, default="R3J-AGNN_model.pkl",
                        help="Path to trained model checkpoint (.pkl).")
    parser.add_argument("--output", type=str, help="Path to save prediction results (optional).")

    # 运行配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda/cpu).")

    return parser.parse_args()


def load_model(model_path, device):
    """加载模型并处理不同的序列化格式"""
    print(f"[*] Loading model from {model_path}...")

    # 1. 初始化模型架构
    model = DualGraphRNAModel(
        micro_in_channels=5,
        macro_node_dim=34,
        macro_edge_dim=11,
        hidden_dim=200,
        dropout=0.50,
        num_gat_layers=5,
        output_dim=6,
        micro_num_layers=3,
        use_edge_features=True,
        use_transformer=False,
        use_global_attn=False,
    ).to(device)

    # 2. 加载权重
    try:
        # 尝试直接使用 torch.load
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        # 如果是 pickle 格式，则手动打开
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

    # 3. 提取 state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print("[+] Model loaded successfully.")
    return model


def parse_rna_fasta(file_path):
    """解析 3 行格式的 FASTA 文件"""
    records = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'r') as f:
        # 过滤空行并去除首尾空格
        lines = [line.strip() for line in f if line.strip()]

    # 每 3 行代表一个条目：>Name, Sequence, Structure
    for i in range(0, len(lines), 3):
        try:
            header, seq, struct = lines[i], lines[i + 1], lines[i + 2]
            if header.startswith('>'):
                name = header[1:].split()[0]
                records.append({'name': name, 'seq': seq, 'struct': struct})
        except IndexError:
            print(f"[Warning] Incomplete record at line {i + 1}. Skipping.")
            break

    print(f"[+] Loaded {len(records)} RNA entries.")
    return records


# 角度转换
def tensor_to_degrees(pred):
    sin_vals = pred[:, 0::2]  # [N, 3]
    cos_vals = pred[:, 1::2]  # [N, 3]

    # 可选：检查 sincos 范围是否符合预期（调试用）
    assert (sin_vals >= 0).all() and (sin_vals <= 1).all(), "sin values out of [0,1]!"
    assert (cos_vals >= -1).all() and (cos_vals <= 1).all(), "cos values out of [-1,1]!"

    angles_rad = torch.atan2(sin_vals, cos_vals)  # range: [0, π]
    angles_deg = torch.rad2deg(angles_rad)         # range: [0, 180]

    return angles_deg


def run_inference(model, data_list, device):
    """执行预测并格式化输出"""
    if not data_list:
        print("[!] No valid data to predict.")
        return

    # 1. 构建 Batch (处理多个 RNA)
    batch_data = Batch.from_data_list(data_list).to(device)

    # 2. 生成用于溯源的 Index (因为 Batch 会把所有节点合并，我们需要知道哪个节点属于哪个图)
    macro_node_counts = [data.macro_x.size(0) for data in data_list]
    macro_batch_idx = torch.cat([
        torch.full((count,), i, dtype=torch.long)
        for i, count in enumerate(macro_node_counts)
    ]).to(device)

    # 3. 预测
    print("[*] Running inference...")
    with torch.no_grad():
        # output 形状为 (Total_Macro_Nodes, 6) -> [sin1, cos1, sin2, cos2, sin3, cos3]
        output = model(batch_data)

        # 4. 筛选 Multi-loop (三叉路口) 节点进行展示
        # 根据 R3J-AGNN 定义，标签 2 通常代表 Junction 核心节点
        labels = batch_data.loop_label
        mask = (labels == 2)

        target_output = output[mask]
        node_indices = torch.nonzero(mask, as_tuple=True)[0]
        graph_indices = macro_batch_idx[node_indices]

    # 5. 打印结果表格
    print("\n" + "=" * 90)
    print(f"{'RNA Name':<25} | {'NodeIdx':<8} | {'Angle Theta1':<12} | {'Angle Theta2':<12} | {'Angle Theta3':<12}")
    print("-" * 90)

    if target_output.size(0) > 0:
        for i, global_idx in enumerate(node_indices):
            graph_idx = graph_indices[i].item()
            rna_name = data_list[graph_idx].rna_name

            # 将 [sin, cos] 转换为 [0, 180] 度的角度值
            raw_pred = target_output[i].view(1, -1)
            angles = tensor_to_degrees(raw_pred).squeeze().tolist()

            print(
                f"{rna_name:<25} | {global_idx.item():<8} | {angles[0]:10.2f}°  | {angles[1]:10.2f}°  | {angles[2]:10.2f}°")
    else:
        print("No Three-way Junction (Multi-loop) nodes detected in the input structure.")
    print("=" * 90 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. 解析 FASTA
    try:
        rna_entries = parse_rna_fasta(args.input)
    except Exception as e:
        print(f"[Error] Failed to parse input: {e}")
        return

    # 2. 数据预处理 (构建双层图结构)
    print("[*] Preprocessing RNA sequences and structures...")
    data_list = []
    for entry in rna_entries:
        # prepare_data 内部会进行树分解和双层图构建
        data = prepare_data(entry['name'], entry['seq'], entry['struct'])
        if data is not None:
            data_list.append(data)

    if not data_list:
        print("[!] Preprocessing failed for all entries.")
        return

    # 3. 加载模型
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # 4. 执行推理并输出结果
    run_inference(model, data_list, device)


if __name__ == "__main__":
    main()

