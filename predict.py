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
# RNA Processing & Decomposition
# ==========================================
def process_rna(rna_name, rna_seq, rna_struct):
    """
    Decompose RNA structure into a Junction Tree and extract node/edge information.
    """
    # Clean sequence: remove extra symbols, convert to uppercase, replace T with U
    processed_seq = rna_seq.replace('&', '').strip().upper().replace('T', 'U')
    processed_struct = rna_struct.replace('&', '').strip()

    # Use lib.tree_decomp (td) for decomposition
    # Note: node_labels is typically a list ['S', 'I', 'M'...]
    try:
        adjmat, node_labels, hpn_nodes_assignment = td.decompose(processed_struct)
    except Exception as e:
        print(f"Error in decomposition for {rna_name}: {e}")
        # Return empty structures to prevent crash
        return [], np.array([[], []]), {}, {}, {}

    # Create RNA Junction Tree object for edge traversal order
    tree = td.RNAJunctionTree(processed_seq, processed_struct)

    # Initialize stack for DFS traversal starting from root (index 0)
    stack = []
    if tree.nodes:
        td.dfs(stack, tree.nodes[0], 0)

    # Extract node information
    node_data = {}
    node_label = {}
    node_bases = {}

    # ================= FIX START =================
    # node_labels is a list, use enumerate to iterate
    for idx, label in enumerate(node_labels):
        # Handle hpn_nodes_assignment whether it is a dict or list
        if isinstance(hpn_nodes_assignment, dict):
            bases = hpn_nodes_assignment.get(idx, [])
        elif isinstance(hpn_nodes_assignment, list):
            bases = hpn_nodes_assignment[idx] if idx < len(hpn_nodes_assignment) else []
        else:
            bases = []

        node_label[idx] = label
        node_bases[idx] = bases
        node_data[idx] = (label, bases)
    # ================= FIX END =================

    # Extract edge information (Forward and Backward edges)
    edges = []
    back_edges = []
    for edge_tuple in stack:
        node1, node2, direction = edge_tuple
        if direction == 1:  # Forward edge
            edges.append((node1.idx, node2.idx))
        else:               # Backward edge
            back_edges.append((node1.idx, node2.idx))

    # Convert to numpy array format for edge_index
    edge_index = np.array(list(zip(*edges))) if edges else np.array([[], []])

    return stack, edge_index, node_label, node_bases, node_data


# ==========================================
# Data Reading Function
# ==========================================
def read_rna_data(file_path):
    """
    Read RNA data from a text file, supporting multi-line reads and various secondary structure symbols.

    Args:
        file_path (str): Path to the RNA data file.

    Returns:
        list: A list of tuples containing (rna_name, rna_seq, rna_struct).
    """

    def is_sequence_line(line):
        """Check if the line contains only base characters."""
        return len(line) > 0 and all(c in 'AUGCaugct& ' for c in line)

    def is_structure_line(line):
        """Check if the line contains only secondary structure symbols."""
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
                # Save the previous RNA entry if it exists
                if current_name is not None and current_seq and current_struct:
                    rna_data.append((current_name, current_seq, current_struct))

                # Reset variables for the next RNA
                current_name = line[1:].strip()
                current_seq, current_struct = "", ""

            elif is_sequence_line(line):
                current_seq += line.replace(' ', '')

            elif is_structure_line(line):
                current_struct += line.replace(' ', '')

    # Don't forget to append the last RNA entry
    if current_name is not None and current_seq and current_struct:
        rna_data.append((current_name, current_seq, current_struct))

    print(f"Successfully loaded {len(rna_data)} RNA records from {file_path}")
    return rna_data


# ==========================================
# RNA Graph Transformation (Stem as Edge Features)
# ==========================================
def transform_rna_graph(rna_name, node_label, edge_index, node_data, node_bases):
    """
    Transform raw RNA graph into a simplified graph: Collapse Stem nodes into edges connecting non-stem nodes.
    """
    stem_nodes = {i for i, lbl in node_label.items() if lbl == 'S'}
    non_stem_nodes = {i for i, lbl in node_label.items() if lbl != 'S'}

    # Create mapping from old indices to new indices
    new_mapping = {old: i for i, old in enumerate(sorted(non_stem_nodes))}

    new_node_data = {new_mapping[o]: node_data[o] for o in non_stem_nodes}
    new_node_bases = {new_mapping[o]: node_bases[o] for o in non_stem_nodes}

    edges = list(zip(edge_index[0], edge_index[1]))
    stem_edges_found = []
    edge_features = {}
    direct_edges = []

    # Process Stem nodes: Convert them into edge information
    for s_node in stem_nodes:
        conn_edges = [e for e in edges if s_node in e]

        # Simplified logic: Find non-stem nodes at both ends of the stem and connect them
        neighbor_nodes = []
        for e in conn_edges:
            neighbor = e[0] if e[0] != s_node else e[1]
            neighbor_nodes.append(neighbor)

        # If the stem connects two non-stem nodes
        for i in range(len(neighbor_nodes)):
            for j in range(i + 1, len(neighbor_nodes)):
                n1, n2 = neighbor_nodes[i], neighbor_nodes[j]
                if n1 not in stem_nodes and n2 not in stem_nodes:
                    new_e = tuple(sorted((n1, n2)))
                    if new_e not in stem_edges_found:
                        stem_edges_found.append(new_e)
                        edge_features[new_e] = {'sequence': node_data[s_node], 'bases': node_bases[s_node]}

    # Retain direct connections between non-stem nodes
    for u, v in edges:
        if u in non_stem_nodes and v in non_stem_nodes:
            direct_edges.append(tuple(sorted((u, v))))

    all_combined_edges = list(set(direct_edges + stem_edges_found))

    # Convert to new index format
    transformed_edges = [(new_mapping[u], new_mapping[v]) for u, v in all_combined_edges]
    new_edge_idx = np.array(list(zip(*transformed_edges))) if transformed_edges else np.array([[], []])

    # Update edge features index
    final_edge_features = {}
    for (u, v), feat in edge_features.items():
        final_edge_features[(new_mapping[u], new_mapping[v])] = feat

    return new_edge_idx, new_mapping, stem_edges_found, final_edge_features, new_node_data, new_node_bases


# ==========================================
# Nucleotide Base Graph Construction
# ==========================================
def parse_base_pairs_with_type(secstruct):
    """
    Parse secondary structure string (supports pseudoknot symbols [] {} <> etc.).
    """
    bps = []
    is_pk = []
    # Parenthesis mapping
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
                        is_pk.append(open_char != '(')  # Anything other than '(' is treated as pseudoknot
                    break
    return bps, is_pk


def get_rna_edge_index(rna_seq, rna_struct):
    """
    Construct the Nucleotide Graph in PyG format.

    Feature Dimension Definition:
    - Node Attr [N, 5]: One-hot (A, U, G, C, N)
    - Edge Attr [E, 4]: One-hot (Watson-Crick, Wobble/Non-canonical, Pseudoknot, Backbone)
    """
    # Preprocessing: Clean and normalize input
    clean_seq = re.sub(r'[^AUCGN]', 'N', rna_seq.upper())
    clean_struct = rna_struct  # Assume already cleaned

    L = len(clean_seq)
    if L == 0:
        return torch.empty(2, 0), torch.empty(0, 4), torch.empty(0, 5)

    # 1. Node Feature Construction
    node_map = {'A': [1, 0, 0, 0, 0], 'U': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0]}
    node_attr = torch.tensor([node_map.get(nt, [0, 0, 0, 0, 1]) for nt in clean_seq], dtype=torch.float)

    # 2. Edge & Feature Construction
    edge_list, edge_attrs = [], []

    def add_bidirectional_edge(i, j, attr):
        edge_list.extend([(i, j), (j, i)])
        edge_attrs.extend([attr, attr])

    # A. Backbone edges
    for i in range(L - 1):
        add_bidirectional_edge(i, i + 1, [0.0, 0.0, 0.0, 1.0])

    # B. Base-pair edges
    bps, pk_flags = parse_base_pairs_with_type(clean_struct)
    canonical = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}

    for (i, j), is_pk in zip(bps, pk_flags):
        if is_pk:
            add_bidirectional_edge(i, j, [0.0, 0.0, 1.0, 0.0])  # Pseudoknot edge
        else:
            pair = (clean_seq[i], clean_seq[j])
            if pair in canonical:
                add_bidirectional_edge(i, j, [1.0, 0.0, 0.0, 0.0])  # Canonical pair
            else:
                add_bidirectional_edge(i, j, [0.0, 1.0, 0.0, 0.0])  # Non-canonical pair

    # Convert to Tensors
    if not edge_list:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, 4)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return edge_index, edge_attr, node_attr


# ==========================================
# Graph Structure Label Construction
# ==========================================
def encode_struct_onehot(micro_struct_str):
    """Convert micro-structure string sequence to One-hot tensor."""
    L = len(micro_struct_str)
    onehot = torch.zeros(L, NUM_STRUCT_TYPES, dtype=torch.float)
    for i, typ in enumerate(micro_struct_str):
        idx = STRUCT_TYPE_TO_IDX.get(typ, STRUCT_TYPE_TO_IDX['X'])
        onehot[i, idx] = 1.0
    return onehot  # (L, NUM_STRUCT_TYPES)


def build_micro_struct_labels(node_data, seq_length):
    """
    Construct a list of micro-structure labels for the entire sequence based on tree graph node information.
    Example: ['S', 'S', 'I', 'I', 'S', 'S', ...]
    """
    micro_struct_str = ['X'] * seq_length

    for struct_id, (struct_type, positions) in node_data.items():
        all_positions = []
        if isinstance(positions, list):
            if not positions: continue
            # Flatten nested lists [[1,2], [3,4]] -> [1,2,3,4]
            if isinstance(positions[0], list):
                for sublist in positions:
                    all_positions.extend(sublist)
            else:
                all_positions = positions
        else:
            continue

        # Map type: Treat P (Pseudoknot) as S (Stem)
        label = 'S' if struct_type == 'P' else struct_type
        if label not in STRUCT_TYPE_TO_IDX: label = 'X'

        for pos in all_positions:
            if isinstance(pos, int) and 0 <= pos < seq_length:
                micro_struct_str[pos] = label

    return micro_struct_str


# ==========================================
# RNA Feature Processor
# ==========================================
class RNAFeatureProcessor:
    """
    Responsible for extracting biological features of nodes in the RNA tree graph.
    """

    def __init__(self):
        # Electron-Ion Interaction Potential (EIIP) for simulating charge distribution
        self.EIIP_dict = {
            'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'U': 0.1335, '-': 0
        }

    # ----------------------------------------
    # Basic & Normalization Logic
    # ----------------------------------------
    def one_hot_node_type(self, node_type, valid_types=('P', 'I', 'B', 'M', 'H')):
        """One-hot encoding for node type."""
        vec = [0] * len(valid_types)
        if node_type in valid_types:
            vec[valid_types.index(node_type)] = 1
        return vec

    def _get_basic_features(self, node_type, degree):
        """Basic topological features."""
        return self.one_hot_node_type(node_type)

    def get_node_dot_bracket(self, rna_struct, node_bases_index):
        """Extract corresponding dot-bracket substring based on indices."""
        if not node_bases_index: return []
        if isinstance(node_bases_index[0], list):  # Nested list
            return [[rna_struct[i] for i in seg] for seg in node_bases_index]
        else:
            return [[rna_struct[i] for i in node_bases_index]]

    def _normalize_data_to_triplet(self, data, node_type, degree, invalid_val='-1'):
        """
        Core Logic: Normalize loop data of any type (sequence/structure/index) into a triplet [L1, L2, L3].
        For Junction (M), retain three branches; for other types, pad or merge.
        """
        if not data: return [invalid_val] * 3

        # Standardization strategy for different node types and degrees
        if node_type == 'M':
            if degree == 3:  # Standard 3-way junction
                if len(data) == 3:
                    return data
                elif len(data) == 4:
                    return [data[0] + data[3], data[1], data[2]]  # Merge head and tail
                elif len(data) > 4:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]  # Merge remaining
            else:  # Non-standard M
                if len(data) <= 3:
                    return data + [invalid_val] * (3 - len(data))
                else:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]

        # 2-way nodes (I, B) -> Pad the third item
        elif node_type in ['I', 'B']:
            if degree == 2:
                if len(data) == 2:
                    return [data[0], data[1], invalid_val]
                elif len(data) > 2:
                    return [data[0], data[1], sum(data[2:], type(data[0])())]
            else:
                return data[:2] + [invalid_val] if len(data) >= 2 else data + [invalid_val] * (3 - len(data))

        # 1-way node (H) -> Merge into one item, pad the last two
        elif node_type == 'H':
            return [sum(data, type(data[0])()), invalid_val, invalid_val]

        # Default handling
        if len(data) == 1:
            return [data[0], invalid_val, invalid_val]
        elif len(data) == 2:
            return [data[0], data[1], invalid_val]
        return [data[0], data[1], sum(data[2:], type(data[0])())]

    def _clean_triplet(self, triplet):
        """Remove '&' connector and handle invalid values."""
        cleaned = []
        for item in triplet:
            if isinstance(item, str):
                s = item.replace('&', '')
                cleaned.append(s if s else '-1')
            elif isinstance(item, list):  # Index list
                cleaned.append(item if item else [])
            else:
                cleaned.append(item)
        return cleaned

    # ----------------------------------------
    # Feature Calculation Sub-functions
    # ----------------------------------------
    def _calc_length_feats(self, bases):
        """Calculate geometric constraint features like length, asymmetry, ratio."""
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
        """Calculate base composition (A/G/C/U/Purine)."""
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

            # Consecutive A motif detection (A-minor motif indicator)
            matches = re.findall('A+', seq)
            feats[f'max_consec_A_{branch}'] = max((len(m) for m in matches), default=0)
        return feats

    def _calc_physicochemical_feats(self, bases):
        """Calculate physicochemical features: U-run (flexibility), Charge gradient."""
        feats = {}
        for i, branch in enumerate(['L1', 'L2', 'L3']):
            seq = bases[i]
            if seq == '-1' or len(seq) < 2:
                feats[f'U_run_{branch}'] = 0
                feats[f'Charge_grad_{branch}'] = 0
                continue

            # U-rich region detection
            u_runs = re.findall('U+', seq)
            feats[f'U_run_{branch}'] = max((len(r) for r in u_runs), default=0)

            # Charge gradient
            eiips = [self.EIIP_dict.get(b, 0) for b in seq]
            feats[f'Charge_grad_{branch}'] = eiips[-1] - eiips[0]
        return feats

    def _encode_flanking(self, seq):
        """Encode flanking bases (related to Stacking energy)."""
        mapping = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
        if not seq or seq == '-1': return 0, 0
        return mapping.get(seq[0], 0), mapping.get(seq[-1], 0)

    # ----------------------------------------
    # Main Processing Function
    # ----------------------------------------
    def _get_junction_features(self, node_type, node_id, G, bases, node_bases_index, dot_bracket):
        """Extract and combine all advanced features."""
        # 1. Preprocessing: Remove closing pairs (first and last base)
        # This focuses only on the single-stranded region inside the Loop
        inner_bases = [b[1:-1] if len(b) > 2 else "" for b in bases]

        # 2. Normalize to triplet
        norm_bases = self._clean_triplet(
            self._normalize_data_to_triplet(inner_bases, node_type, G.degree(node_id), '-1')
        )

        # 3. Calculate various features
        f_len = self._calc_length_feats(norm_bases)
        f_comp = self._calc_composition_feats(norm_bases)
        f_phys = self._calc_physicochemical_feats(norm_bases)

        # 4. Flanking base features
        flank_feats = []
        for i, seq in enumerate(norm_bases):
            s, e = self._encode_flanking(seq)
            flank_feats.extend([s, e])

        # 5. Assemble feature vector (Maintain fixed order)
        # Lengths (12) + Composition (21) + Phys (6) + Flanking (6) + Sorted (4) ...
        # Must align with model input dimensions
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
        """External Interface: Calculate complete feature vector for a specified node."""
        # Get dot-bracket representation
        dot_bracket = self.get_node_dot_bracket(rna_struct, node_bases_index[node_id])

        # Correct node type logic
        degree = G.degree(node_id)
        stem_conns = sum(1 for edge in edge_data.keys() if node_id in edge)

        effective_type = node_type
        # If M but only 2 Stem connections, treat as I (Internal Loop)
        if node_type == 'M' and stem_conns == 2:
            effective_type = 'I'
        elif degree == 3 and len(bases) >= 4:
            effective_type = 'I'

        # Calculate features
        # basic_vec = self._get_basic_features(effective_type, degree)
        junc_vec = self._get_junction_features(effective_type, node_id, G, bases, node_bases_index[node_id],
                                               dot_bracket)

        return np.array(junc_vec, dtype=np.float32)


# ==========================================
# Helper Calculations
# ==========================================
def get_max_consecutive_length(indices):
    """Calculate the length of the longest consecutive subsequence in a list of indices."""
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
# Edge Feature Calculation (Stem Features)
# ==========================================
def calculate_edge_features(rna_name, edge_data, rna_seq):
    """
    Calculate features for the edge (Stem) connecting two macro nodes.
    """
    features = {
        'stem_length': 0, 'gc_pairs': 0, 'au_pairs': 0, 'gu_pairs': 0,
        'gc_ratio': 0, 'au_ratio': 0, 'gu_ratio': 0,
        'hydrogen_bonds': 0,
        'consecutive_gc': 0, 'consecutive_au': 0, 'consecutive_gu': 0,
    }

    bases_data = edge_data.get('bases', [[], []])
    # If data is empty, return an 11-dimensional zero vector
    if not bases_data or not bases_data[0]:
        return [0.0] * 11 

    # Process Stem data format
    if isinstance(bases_data[0], list):
        stem_length = len(bases_data[0])
        idx_list1 = bases_data[0]
        idx_list2 = bases_data[1]
    else: 
        stem_length = 1
        idx_list1 = [bases_data[0]]
        idx_list2 = [bases_data[1]]
    
    features['stem_length'] = stem_length

    # Extract sequences and pair them
    try:
        seq1 = [rna_seq[i].upper() for i in idx_list1]
        seq2 = [rna_seq[i].upper() for i in idx_list2]
        seq2 = seq2[::-1]  # Reverse to match pairs
    except IndexError:
        print(f"Warning: Index out of range in edge calculation for {rna_name}")
        return [0.0] * 11

    # Count pair types
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

    # === Return Features (11 dimensions) ===
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
    """Batch process all edge features."""
    # Ensure edge order matches edge_index
    # edge_index: [2, E] tensor
    num_edges = edge_index.size(1)
    edge_features = []

    # Build lookup table: (u, v) -> feature_data
    data_lookup = {}
    for (u, v), data in edge_data_dict.items():
        data_lookup[(u, v)] = data
        data_lookup[(v, u)] = data

    for i in range(num_edges):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if (u, v) in data_lookup:
            feat = calculate_edge_features(rna_name, data_lookup[(u, v)], rna_seq)
        else:
            feat = [0.0] * 11 # Fixed: must match 11 dimensions
        edge_features.append(feat)

    return torch.tensor(edge_features, dtype=torch.float), edge_index


# ==========================================
# Node Feature Calculation (Loop Features)
# ==========================================
def calculate_node_features(rna_name, rna_struct, node_data, node_bases_index, edge_index, edge_data, rna_seq):
    """
    Calculate feature matrices for all nodes.
    """
    feature_processor = RNAFeatureProcessor()

    # Construct NetworkX graph for calculating degree
    G = nx.Graph()
    G.add_nodes_from(node_data.keys())
    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        G.add_edges_from(zip(src, dst))
    else:  # Compatible with numpy
        G.add_edges_from(zip(edge_index[0], edge_index[1]))

    node_ids = sorted(node_data.keys())
    features_list = []

    # Helper: Get sequence segments
    def get_seqs(indices):
        if not indices: return []
        if isinstance(indices[0], list):  # Nested list
            return [''.join([rna_seq[i] for i in seg]) for seg in indices]
        else:  # Flat list
            return [''.join([rna_seq[i] for i in indices])]

    for node_id in node_ids:
        try:
            node_type, _ = node_data[node_id]  # node_data: {id: (type, indices_raw)}
            indices = node_bases_index[node_id]
            bases_seqs = get_seqs(indices)

            # Call Processor
            # process_structure(self, rna_name, rna_struct, node_data, node_type, bases, node_id, G, node_bases_index, edge_data)
            feat_vec = feature_processor.process_structure(
                rna_name=rna_name,
                rna_struct=rna_struct,
                node_data=node_data,
                node_type=node_type,
                bases=bases_seqs,
                node_id=node_id,
                G=G,
                node_bases_index=node_bases_index,  # Pass complete dictionary
                edge_data=edge_data  # Pass complete edge dictionary for connection count
            )
            features_list.append(feat_vec)

        except Exception as e:
            print(f"Error processing node {node_id} in {rna_name}: {e}")
            features_list.append(np.zeros(34, dtype=np.float32))

    if not features_list:
        return torch.tensor([], dtype=torch.float)

    return torch.tensor(np.array(features_list), dtype=torch.float)


# ==========================================
# Custom Graph Data Wrapper
# ==========================================
class RNADualGraphData(Data):
    """
    Core data structure for R3J-AGNN, containing graph information at both nucleotide and tree graph levels.
    Inherits from torch_geometric.data.Data to support automatic Batching.
    """

    def __init__(self,
                 # === Micro-Graph (Atomic/Nucleotide Level) ===
                 micro_x=None,  # [L, 5]: Node features (One-hot Base)
                 micro_struct_attr=None,  # [L, 7]: Structure type features (One-hot Loop Type)
                 micro_edge_index=None,  # [2, E_micro]: Edge indices
                 micro_edge_attr=None,  # [E_micro, 4]: Edge features (Pair Type)

                 # === Macro-Graph (Motif/Topology Level) ===
                 macro_x=None,  # [N_macro, F_node]: Node features
                 macro_edge_index=None,  # [2, E_macro]: Edge indices
                 macro_edge_attr=None,  # [E_macro, F_edge]: Edge features

                 loop_label=None,  # [N_macro]: Node type labels (for Loss Masking)

                 # === Cross-Level Mapping ===
                 micro_to_macro=None,  # [L]: Macro node/edge index for each nucleotide

                 # === Metadata ===
                 rna_name=None,
                 rna_seq=None,
                 rna_struct=None,
                 seq_length=None,
                 y=None,  # [N_macro, 3/6]: Prediction targets (angles)

                 **kwargs):
        super().__init__(**kwargs)

        # Assignments
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
        Key logic for PyG Batching: Define increments for each attribute during concatenation.
        """
        if key == 'micro_edge_index':
            return self.seq_length

        if key == 'macro_edge_index':
            return self.macro_x.size(0) if self.macro_x is not None else 0

        if key == 'micro_to_macro':
            return self.macro_x.size(0) if self.macro_x is not None else 0

        return super().__inc__(key, value, *args, **kwargs)


# ==========================================
# 17. Build Mapping (Assignment Matrix)
# ==========================================
def create_micro_to_macro_mapping(
        new_node_data,       # {macro_idx: (type, bases_list)}
        original_node_bases, # {old_tree_id: (type, bases_list)}
        new_mapping,         # {old_tree_id -> macro_idx}
        seq_length,
        original_edge_index  # Original junction tree edge indices
):
    """
    Construct the micro_to_macro tensor, assigning each nucleotide to a node or edge in the tree graph.
    - Positive (>=0): Index of a Macro Node
    - Negative (<0):  Index of a Macro Edge (for Stem)
    - -100:           Unassigned/Invalid
    """
    # Initialize mapping tensor
    mapping = torch.full((seq_length,), -100, dtype=torch.long)

    def flatten_indices(obj):
        """Recursively flatten index lists."""
        if isinstance(obj, int): return [obj]
        res = []
        for item in obj:
            res.extend(flatten_indices(item))
        return res

    # 1. Map Macro Nodes (Loop/Junction)
    for macro_idx, (n_type, bases) in new_node_data.items():
        if not bases: continue
        for idx in flatten_indices(bases):
            if 0 <= idx < seq_length:
                mapping[idx] = macro_idx

    # 2. Build adjacency list for the original tree (to find neighbors of Stems)
    adj = defaultdict(list)
    if isinstance(original_edge_index, np.ndarray):
        edge_iter = zip(original_edge_index[0], original_edge_index[1])
    elif torch.is_tensor(original_edge_index):
        edge_iter = zip(original_edge_index[0].tolist(), original_edge_index[1].tolist())
    else:
        edge_iter = []
        
    for u, v in edge_iter:
        adj[u].append(v)
        adj[v].append(u)

    # 3. Map Macro Edges (Stem)
    stem_edge_map = {} 
    stem_counter = 0

    for old_id, (n_type, bases) in original_node_bases.items():
        if old_id in new_mapping: continue  # Skip processed Loop nodes
        
        # This is a Stem node, find its connected Loops
        neighbors = [n for n in adj[old_id] if n in new_mapping]
        
        # A Stem is valid only if it connects two valid Loops
        if len(neighbors) == 2:
            macro_u = new_mapping[neighbors[0]]
            macro_v = new_mapping[neighbors[1]]
            
            if macro_u > macro_v: 
                macro_u, macro_v = macro_v, macro_u

            # Use negative indices to mark edges: -1, -2, -3...
            stem_key = -(stem_counter + 1)
            
            for idx in flatten_indices(bases):
                if 0 <= idx < seq_length and mapping[idx] == -100: 
                    mapping[idx] = stem_key
            
            stem_edge_map[stem_key] = (macro_u, macro_v)
            stem_counter += 1

    return mapping, stem_edge_map


# ==========================================
# Main Inference Pipeline (prepare_data)
# ==========================================
def prepare_data(rna_name, rna_seq, rna_struct):
    """
    Single-sample inference data preparation pipeline:
    1. Parse sequence and structure
    2. Generate Micro-Graph (Base Graph)
    3. Generate Macro-Graph (Motif Graph)
    4. Extract dual-level features
    5. Encapsulate into PyG Data object
    """
    try:
        # Preprocessing: Clean sequence
        clean_seq = re.sub(r'[^AUCGaucg]', '', rna_seq).upper()
        # Note: Assume rna_struct length matches clean_seq, otherwise handle mismatch
        if len(clean_seq) != len(rna_struct):
             # Simple alignment attempt (reference only, depends on data source)
             clean_seq = rna_seq.replace('&', '').upper()
             rna_struct = rna_struct.replace('&', '')
        
        seq_length = len(clean_seq)
        if seq_length == 0: return None

        # ------------------------------------
        # Step 1: Original Junction Tree Decomposition
        # ------------------------------------
        stack, edge_index, node_label, node_bases, node_data = process_rna(rna_name, clean_seq, rna_struct)

        # ------------------------------------
        # Step 2: Macro-Graph Transformation (Collapse Stems)
        # ------------------------------------
        new_edge_index, new_mapping, stem_edges, edge_data, new_node_data, new_node_bases_index = \
            transform_rna_graph(rna_name, node_label, edge_index, node_data, node_bases)

        # ------------------------------------
        # Step 3: Micro-Graph Construction (Atomic Graph)
        # ------------------------------------
        # Get base pairs and backbone connections
        secstruct_edge_indices, micro_edge_attr, micro_x = get_rna_edge_index(clean_seq, rna_struct)
        
        # Convert micro-graph indices to Tensor
        # Fixed: Use clone().detach() or direct conversion to avoid warnings
        if isinstance(secstruct_edge_indices, torch.Tensor):
            micro_edge_index = secstruct_edge_indices.clone().detach().long()
        else:
            micro_edge_index = torch.tensor(secstruct_edge_indices, dtype=torch.long)
        
        # Construct structural labels for micro-nodes (S, H, I, M...)
        micro_struct_str = build_micro_struct_labels(new_node_data, seq_length)
        micro_struct_attr = encode_struct_onehot(micro_struct_str)

        # ------------------------------------
        # Step 4: Macro Feature Extraction
        # ------------------------------------
        # 4.1 Edge Features (Stem properties)
        edge_index_tensor = torch.tensor(new_edge_index, dtype=torch.long)
        macro_edge_attr, _ = process_all_edges(rna_name, edge_data, edge_index_tensor, clean_seq)

        # 4.2 Node Features (Loop properties)
        # Note: In inference mode without PDB, skip geometric correction, use topological info directly
        macro_x = calculate_node_features(
            rna_name, rna_struct, new_node_data, new_node_bases_index, 
            new_edge_index, edge_data, clean_seq
        )
        
        # ------------------------------------
        # Step 5: Labels and Mapping
        # ------------------------------------
        # Generate Loop Labels (H=0, I=1, M=2...)
        node_type_map = {'H': 0, 'I': 1, 'M': 2, 'P': 3, 'S': 4, 'E': 5, 'B': 6, 'X': 7}
        sorted_node_ids = sorted(new_node_data.keys())
        loop_label_list = [node_type_map.get(new_node_data[nid][0], 7) for nid in sorted_node_ids]
        loop_label = torch.tensor(loop_label_list, dtype=torch.long)

        # Generate Micro-Macro Mapping
        micro_to_macro, _ = create_micro_to_macro_mapping(
            new_node_data, node_data, new_mapping, seq_length, torch.tensor(edge_index)
        )

        # ------------------------------------
        # Step 6: Encapsulation
        # ------------------------------------
        data = RNADualGraphData(
            # Micro-Graph
            micro_x=micro_x,
            micro_struct_attr=micro_struct_attr,
            micro_edge_index=micro_edge_index,
            micro_edge_attr=micro_edge_attr,

            # Macro-Graph
            macro_x=macro_x,
            macro_edge_index=edge_index_tensor,
            macro_edge_attr=macro_edge_attr,
            loop_label=loop_label,

            # Correlation
            micro_to_macro=micro_to_macro,

            # Metadata
            rna_name=rna_name,
            rna_seq=rna_seq,
            rna_struct=rna_struct,
            seq_length=seq_length,

            # Empty labels for inference
            y=None
        )

        return data

    except Exception as e:
        print(f"Error processing inference data {rna_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# Parse Custom FASTA File
# ==========================================
def parse_rna_fasta(file_path):
    records = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return records

    with open(file_path, 'r') as f:
        # Filter empty lines and strip whitespace
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
                # Extract name, remove '>' and potential parameters
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