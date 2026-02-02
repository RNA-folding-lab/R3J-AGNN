# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout, Embedding

# PyTorch Geometric imports
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import GATConv, TransformerConv


def scatter_softmax(src, index, dim=0, eps=1e-8):
    """Grouped softmax for scatter operations."""
    max_val = scatter(src, index, dim=dim, reduce='max')
    src = src - max_val[index]
    out = src.exp()
    out_sum = scatter(out, index, dim=dim, reduce='sum') + eps
    return out / out_sum[index]


class GlobalSelfAttention(nn.Module):
    """Global self-attention mechanism with learnable positional embeddings."""

    def __init__(self, embed_dim, num_heads=2, dropout=0.2, max_seq_len=512):
        super().__init__()
        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)
        self.pos_embed = Embedding(max_seq_len, embed_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch=batch)  # (B, L, D), (B, L)
        B, L, D = x_dense.shape

        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}")

        pos_ids = torch.arange(L, device=x_dense.device).unsqueeze(0)
        pos_enc = self.pos_embed(pos_ids)

        x_with_pos = x_dense + pos_enc
        attn_out, _ = self.attn(
            x_with_pos, x_with_pos, x_with_pos,
            key_padding_mask=~mask
        )
        attn_out = self.dropout(attn_out)
        x_dense = self.norm(x_dense + attn_out)
        return x_dense[mask]


class NodeFeatureEnhancer(nn.Module):
    """Enriches node features with sinusoidal positional encoding and 1D-CNN context."""

    def __init__(self, input_dim, hidden_dim, use_pos_enc=True, use_cnn_context=True):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.use_cnn_context = use_cnn_context
        self.base_proj = nn.Linear(input_dim, hidden_dim)

        if use_pos_enc:
            self.pos_scale = nn.Parameter(torch.ones(1) * 0.1)

        if use_cnn_context:
            self.context_conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )
            self.norm_cnn = nn.LayerNorm(hidden_dim)

    def get_sinusoidal_encoding(self, seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, batch=None):
        x_emb = self.base_proj(x)
        if self.use_pos_enc:
            L, D = x_emb.shape
            pe = self.get_sinusoidal_encoding(L, D, x.device)
            x_emb = x_emb + pe * self.pos_scale

        if self.use_cnn_context:
            x_cnn = x_emb.unsqueeze(0).transpose(1, 2)
            x_cnn = self.context_conv(x_cnn)
            x_cnn = x_cnn.transpose(1, 2).squeeze(0)
            x_emb = x_emb + self.norm_cnn(F.gelu(x_cnn))
        return x_emb


class MicroGNN(nn.Module):
    """Low-level GNN to process residue-level graph information."""

    def __init__(self, node_feat_dim=5, struct_feat_dim=7, edge_feat_dim=4,
                 hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        total_input_dim = node_feat_dim + struct_feat_dim
        self.feat_enhancer = NodeFeatureEnhancer(total_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // 2,
                    heads=2,
                    edge_dim=edge_feat_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, micro_x, micro_struct_attr, micro_edge_index, micro_edge_attr, batch=None):
        raw_x = torch.cat([micro_x, micro_struct_attr], dim=-1)
        x = self.feat_enhancer(raw_x, batch)

        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            x_in = x
            x = conv(x, micro_edge_index, edge_attr=micro_edge_attr)
            x = norm(x + x_in)
            x = F.gelu(x)
            x = drop(x)
        return x


class AttentiveMicroAggregator(nn.Module):
    """Aggregates micro-level features to macro-level nodes/edges using attention."""

    def __init__(self, micro_dim, macro_dim, dropout=0.1):
        super().__init__()
        self.micro_dim = micro_dim
        self.attn_lin = nn.Linear(micro_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(micro_dim, macro_dim),
            nn.LayerNorm(macro_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _attentive_aggregate(self, micro_feat, indices, num_targets):
        if micro_feat.size(0) == 0:
            return torch.zeros(num_targets, self.micro_dim, device=micro_feat.device)
        attn_logits = self.attn_lin(micro_feat).squeeze(-1)
        attn_weights = scatter_softmax(attn_logits, indices, dim=0)
        weighted = micro_feat * attn_weights.unsqueeze(-1)
        return scatter(weighted, indices, dim=0, dim_size=num_targets, reduce='sum')

    def forward(self, micro_features, micro_to_macro, num_macro_nodes, num_macro_edges):
        node_mask = micro_to_macro >= 0
        edge_mask = (micro_to_macro < 0) & (micro_to_macro != -100)

        node_agg = self._attentive_aggregate(micro_features[node_mask], micro_to_macro[node_mask], num_macro_nodes) \
            if node_mask.any() else torch.zeros(num_macro_nodes, self.micro_dim, device=micro_features.device)

        edge_agg = self._attentive_aggregate(micro_features[edge_mask], -micro_to_macro[edge_mask] - 1, num_macro_edges) \
            if edge_mask.any() else torch.zeros(num_macro_edges, self.micro_dim, device=micro_features.device)

        return self.output_proj(node_agg), self.output_proj(edge_agg)


class GatedFusion(nn.Module):
    """Symmetric gated fusion for combining micro and macro features."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_b):
        gate = self.gate_net(torch.cat([feat_a, feat_b], dim=-1))
        return self.norm(gate * feat_a + (1 - gate) * feat_b)


class DualGraphRNAModel(nn.Module):
    """Main Dual-Graph RNA model for geometry prediction."""

    def __init__(self, micro_in_channels=5, macro_node_dim=34, macro_edge_dim=11,
                 hidden_dim=200, dropout=0.5, num_gat_layers=5, output_dim=6,
                 micro_num_layers=3, use_edge_features=True,
                 use_transformer=False, use_global_attn=False):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.use_transformer = use_transformer
        self.use_global_attn = use_global_attn

        self.micro_encoder = MicroGNN(micro_in_channels, hidden_dim=hidden_dim,
                                      num_layers=micro_num_layers, dropout=dropout)
        self.micro_aggregator = AttentiveMicroAggregator(hidden_dim, hidden_dim, dropout)

        self.macro_node_encoder = nn.Sequential(
            nn.Linear(macro_node_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )

        if use_edge_features:
            self.macro_edge_encoder = nn.Sequential(
                nn.Linear(macro_edge_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout / 2)
            )

        self.node_fusion = GatedFusion(hidden_dim, dropout)
        self.edge_fusion = GatedFusion(hidden_dim, dropout)

        self.gat_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads=2,
                                                 edge_dim=hidden_dim if use_edge_features else None,
                                                 dropout=dropout, concat=False, add_self_loops=True)
                                         for _ in range(num_gat_layers)])
        self.prenorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])
        self.postnorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])

        if use_global_attn:
            self.global_attn = GlobalSelfAttention(hidden_dim, num_heads=max(1, hidden_dim // 32), dropout=dropout)

        if use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 4,
                                           dropout=dropout, batch_first=True, activation='gelu'),
                num_layers=1
            )
            self.pos_embed = nn.Embedding(512, hidden_dim)

        self.angle_heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0);
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        micro_feat = self.micro_encoder(data.micro_x, data.micro_struct_attr,
                                        data.micro_edge_index, data.micro_edge_attr)

        agg_node_m, agg_edge_m = self.micro_aggregator(
            micro_feat, data.micro_to_macro,
            data.macro_x.size(0),
            data.macro_edge_attr.size(0) if hasattr(data, 'macro_edge_attr') else 0
        )

        m_node_feat = self.macro_node_encoder(data.macro_x)
        m_edge_feat = self.macro_edge_encoder(data.macro_edge_attr) if self.use_edge_features else None

        x = self.node_fusion(m_node_feat, agg_node_m)
        edge_feat = self.edge_fusion(m_edge_feat, agg_edge_m) if m_edge_feat is not None else agg_edge_m

        for pre, gat, post in zip(self.prenorms, self.gat_layers, self.postnorms):
            x = post(x + gat(pre(x), data.macro_edge_index, edge_attr=edge_feat))

        if self.use_transformer:
            batch_vec = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0),
                                                                                                         dtype=torch.long,
                                                                                                         device=x.device)
            x_dense, mask = to_dense_batch(x, batch=batch_vec)
            pos_ids = torch.arange(x_dense.size(1), device=x.device).unsqueeze(0)
            x_dense = self.transformer(x_dense + self.pos_embed(pos_ids), src_key_padding_mask=~mask)
            x = x_dense[mask]

        if self.use_global_attn and hasattr(data, 'batch'):
            x = self.global_attn(x, data.batch)

        out_raw = self.angle_heads(x)
        return torch.stack([torch.sigmoid(out_raw[:, ::2]), torch.tanh(out_raw[:, 1::2])], dim=-1).view(-1, 6)
