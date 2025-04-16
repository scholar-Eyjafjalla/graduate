import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
import numpy as np

from diffab.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from diffab.modules.common.layers import mask_zero, LayerNorm
from diffab.utils.protein.constants import BBHeavyAtom, AA, backbone_atom_coordinates_tensor


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated

class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt) 
            for _ in range(num_layers)
        ])

    def forward(self, R, t, res_feat, pair_feat, mask):
        for i, block in enumerate(self.blocks):
            res_feat = block(R, t, res_feat, pair_feat, mask)
        return res_feat


import sys
sys.path.append("/home/hkm/antibody/dyMEAN")  # 添加dyMEAN的根目录
# from models.dyMEAN.dyMEAN_model import AMEncoder
# # from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, EdgeConstructor


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, *, *)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)  # (N, L, L, *)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)

# import torch
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_add

class EGNNLayer(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128):
        super().__init__()
        self.coord_net = nn.Sequential(
            nn.Linear(node_feat_dim*2 + edge_feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.node_net = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )

    def forward(self, x, edge_index, edge_attr, pos, mask=None):
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # 坐标更新
        coord_feat = torch.cat([x[src], x[dst], edge_attr, rel_dist], dim=-1)
        coord_weights = self.coord_net(coord_feat)
        delta_coord = coord_weights * rel_pos
        
        # 应用掩码过滤无效边
        if mask is not None:
            valid_edge = mask[src] & mask[dst]
            delta_coord = delta_coord * valid_edge.float().unsqueeze(-1)
        
        agg_delta = scatter_mean(delta_coord, dst, dim=0, dim_size=x.size(0))
        new_pos = pos + agg_delta
        
        # 节点特征更新
        agg_edge = scatter_mean(edge_attr, dst, dim=0, dim_size=x.size(0))
        node_feat = torch.cat([x, agg_edge], dim=-1)
        new_x = self.node_net(node_feat)
        
        return new_x, new_pos

class GNNEncoder(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            EGNNLayer(node_feat_dim, pair_feat_dim + 1, hidden_dim=128)
            for _ in range(num_layers)
        ])

    def forward(self, R, t, res_feat, pair_feat, mask_res):
        B, L = mask_res.shape
        device = mask_res.device
        
        # 获取 Cα 坐标并展平 [B, L, 3] -> [B*L, 3]
        ca_coord = t.view(B, L, 3).reshape(B*L, 3)
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), L)
        
        # 构建 KNN 图
        edge_index = knn_graph(
            x=ca_coord,
            k=30,
            batch=batch_idx,
            loop=False
        )
        
        # 提取边特征 [B, L, L, D] -> [num_edges, D]
        edge_pair_feat = pair_feat[batch_idx[edge_index[0]], edge_index[0] % L, edge_index[1] % L]
        rel_pos = ca_coord[edge_index[0]] - ca_coord[edge_index[1]]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([edge_pair_feat, rel_dist], dim=-1)
        
        # 处理掩码 [B, L] -> [B*L]
        valid_mask = mask_res.view(-1)
        
        # 消息传递
        node_feat = res_feat.view(B*L, -1)
        for layer in self.gnn_layers:
            node_feat, ca_coord = layer(
                x=node_feat,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=ca_coord,
                mask=valid_mask
            )
        
        # 重建 C 和 N 的坐标
        R_flat = R.view(B*L, 3, 3)
        c_coord = ca_coord + R_flat[..., 2]  # 沿 z 轴（C 方向）
        n_coord = ca_coord + R_flat[..., 0]  # 沿 x 轴（N 方向）
        
        # 重建旋转矩阵
        new_R = construct_3d_basis(
            center=ca_coord,
            p1=c_coord,
            p2=n_coord
        ).view(B, L, 3, 3)
        
        # 更新平移向量
        new_t = ca_coord.view(B, L, 3)
        
        return node_feat.view(B, L, -1)
    
    
def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res

def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res

CONSTANT = 1
NUM_SEG = 1  # if you do not have enough memory or you have large attr_size, increase this parameter

def coord2radial(edge_index, coord, attr, channel_weights, linear_map):
    '''
    :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
    :param coord: [N, n_channel, d]
    :param attr: [N, n_channel, attr_size], attribute embedding of each channel
    :param channel_weights: [N, n_channel], weights of different channels
    :param linear_map: nn.Linear, map features to d_out
    :param num_seg: split row/col into segments to reduce memory cost
    '''
    row, col = edge_index
    N, n_channel, d = coord.shape

    radials = []

    seg_size = (len(row) + NUM_SEG - 1) // NUM_SEG


    assert torch.all(row >= 0) and torch.all(row < N), f"Invalid row indices: min {row.min()}, max {row.max()} vs N={N}"
    assert torch.all(col >= 0) and torch.all(col < N), f"Invalid col indices: min {col.min()}, max {col.max()} vs N={N}"
    
    for i in range(NUM_SEG):
        start = i * seg_size
        end = min(start + seg_size, len(row))
        if end <= start:
            break
        seg_row, seg_col = row[start:end], col[start:end]

        coord_msg = torch.norm(
            coord[seg_row].unsqueeze(2) - coord[seg_col].unsqueeze(1),  # [n_edge, n_channel, n_channel, d]
            dim=-1, keepdim=False)  # [n_edge, n_channel, n_channel]
        
        # print("[DEBUG]: coord_msg shape", coord_msg.shape)
        # print("[DEBUG]: channel_weights shape", channel_weights.shape)
        # print("[DEBUG]: channel_weights[seg_row] shape", channel_weights[seg_row].shape)
        # print("[DEBUG]: channel_weights[seg_col] shape", channel_weights[seg_col].shape)

        coord_msg = coord_msg * torch.bmm(
            channel_weights[seg_row].unsqueeze(2),
            channel_weights[seg_col].unsqueeze(1)
            )  # [n_edge, n_channel, n_channel]
        
        radial = torch.bmm(
            attr[seg_row].transpose(-1, -2),  # [n_edge, attr_size, n_channel]
            coord_msg)  # [n_edge, attr_size, n_channel]
        radial = torch.bmm(radial, attr[seg_col])  # [n_edge, attr_size, attr_size]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, attr_size * attr_size]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT  # post norm
        radial = linear_map(radial) / radial_norm # [n_edge, d_out]

        radials.append(radial)
    
    radials = torch.cat(radials, dim=0)  # [N_edge, d_out]

    # generate coord_diff by first mean src then minused by dst
    # message passed from col to row
    channel_mask = (channel_weights != 0).long()  # [N, n_channel]
    channel_sum = channel_mask.sum(-1)  # [N]
    pooled_col_coord = (coord[col] * channel_mask[col].unsqueeze(-1)).sum(1)  # [n_edge, d]
    pooled_col_coord = pooled_col_coord / channel_sum[col].unsqueeze(-1)  # [n_edge, d], denominator cannot be 0 since no pad node exists
    coord_diff = coord[row] - pooled_col_coord.unsqueeze(1)  # [n_edge, n_channel, d]

    return radials, coord_diff

# @singleton
class RollerPooling(nn.Module):
    '''
    Adaptive average pooling for the adaptive scaler
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.n_channel = n_channel
        with torch.no_grad():
            pool_matrix = []
            ones = torch.ones((n_channel, n_channel), dtype=torch.float)
            for i in range(n_channel):
                # i start from 0 instead of 1 !!! (less readable but higher implemetation efficiency)
                window_size = n_channel - i
                mat = torch.triu(ones) - torch.triu(ones, diagonal=window_size)
                pool_matrix.append(mat / window_size)
            self.pool_matrix = torch.stack(pool_matrix)
    
    def forward(self, hidden, target_size):
        '''
        :param hidden: [n_edges, n_channel]
        :param target_size: [n_edges]
        '''
        pool_mat = self.pool_matrix.to(hidden.device).type(hidden.dtype)
        target_size = target_size.to(torch.long) - 1
        pool_mat = pool_mat[target_size]  # [n_edges, n_channel, n_channel]
        hidden = hidden.unsqueeze(-1)  # [n_edges, n_channel, 1]
        return torch.bmm(pool_mat, hidden)  # [n_edges, n_channel, 1]

# def coord2radial(edge_index, coord, attr, channel_weights, linear_map):
#     """ 直接从dyMEAN的am_egnn.py中复制的完整实现 """
#     row, col = edge_index
#     N, n_channel, d = coord.shape
#     radials = []
#     seg_size = (len(row) + 3) // 4  # 原版分4段处理
    
#     for i in range(4):
#         start = i * seg_size
#         end = min(start + seg_size, len(row))
#         if end <= start: break
        
#         seg_row, seg_col = row[start:end], col[start:end]
#         coord_i = coord[seg_row]  # [seg, C, 3]
#         coord_j = coord[seg_col]  # [seg, C, 3]
        
#         # 跨通道距离矩阵 [seg, C, C]
#         coord_msg = torch.norm(coord_i.unsqueeze(2) - coord_j.unsqueeze(1), dim=-1)
#         coord_msg *= torch.bmm(
#             channel_weights[seg_row].unsqueeze(2), 
#             channel_weights[seg_col].unsqueeze(1)
#         )  # 通道权重交互
        
#         # 属性交互 [seg, D, D]
#         radial = torch.bmm(
#             attr[seg_row].transpose(1,2),  # [seg, D, C]
#             coord_msg
#         )  # [seg, D, C]
#         radial = torch.bmm(radial, attr[seg_col])  # [seg, D, D]
#         radial = radial.reshape(radial.size(0), -1)  # [seg, D*D]
        
#         # 归一化
#         radial_norm = torch.norm(radial, dim=1, keepdim=True) + 1e-5
#         radial = linear_map(radial) / radial_norm
        
#         radials.append(radial)
    
#     radial = torch.cat(radials, dim=0)  # [E, D]
    
#     # 生成coord_diff
#     channel_sum = (channel_weights != 0).sum(-1)  # [N]
#     pooled_j = (coord[col] * channel_weights[col].unsqueeze(-1)).sum(1)  # [E,3]
#     pooled_j = pooled_j / channel_sum[col].unsqueeze(-1)
#     coord_diff = coord[row] - pooled_j.unsqueeze(1)  # [E, C, 3]
    
#     return radial, coord_diff

class NewGNNEncoder(nn.Module):
    def __init__(self, 
                 node_feat_dim: int,
                 num_atom_types: int = 5, 
                 num_atom_pos: int = 6,
                 hidden_dim: int = 128,
                 n_channel: int = 4,
                 radial_dim: int = 256,
                 num_layers: int = 4,
                 k_neighbors: int = 30,
                 dropout: float = 0.1):
        super().__init__()
        self.n_channel = n_channel
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        
        # 原子类型/位置嵌入（仿dyMEAN的AminoAcidEmbedding）
        self.atom_type_embed = nn.Embedding(num_atom_types, hidden_dim//2)
        self.atom_pos_embed = nn.Embedding(num_atom_pos, hidden_dim//2)
        
        # 原子特征投影（输出动态权重）
        self.atom_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim + n_channel),
            nn.SiLU()
        )
        
        # 径向基映射层（与dyMEAN一致）
        self.radial_linear = nn.Linear((hidden_dim//2 + 3)**2, radial_dim)  # 根据属性维度计算
        
        # GNN层
        self.ctx_layers = nn.ModuleList([
            AM_E_GCL(
                input_nf=hidden_dim,
                hidden_nf=hidden_dim,
                output_nf=hidden_dim,
                n_channel=n_channel,
                channel_nf=hidden_dim//2 + 3,
                radial_nf=radial_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.inter_layers = nn.ModuleList([
            AM_E_GCL(
                input_nf=hidden_dim,
                hidden_nf=hidden_dim,
                output_nf=hidden_dim,
                n_channel=n_channel,
                channel_nf=hidden_dim//2 + 3,
                radial_nf=radial_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.fusion_gate = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 坐标标准化（与dyMEAN一致）
        self.normalizer = SeperatedCoordNormalizer()

    def _build_edges(self, X, segment_ids, batch_id):
        """构建k近邻边，区分链内和链间边"""
        # B, L = X.shape[:2]
        # X_flat = X.view(B*L, self.n_channel, 3)
        X_flat = X  # 统一为 [N, 5, 3]
        # 分批次处理
        edges = []
        for bid in torch.unique(batch_id):
            mask = (batch_id == bid)
            X_batch = X_flat[mask]  # [N_bid, 5, 3]
            seg_batch = segment_ids.view(-1)[mask]

            # 计算距离矩阵
            dist = torch.cdist(X_batch[:,1], X_batch[:,1])  # 使用CA原子距离
            dist_mask = seg_batch[:,None] == seg_batch[None,:]  # 同链掩码

            # 链内边
            intra_edges = self._knn_with_mask(dist, dist_mask, self.k_neighbors)
            # 链间边
            inter_edges = self._knn_with_mask(dist, ~dist_mask, self.k_neighbors//3)

            ag_mask = (seg_batch == 0)  # 抗原节点
            ab_mask = (seg_batch > 0)   # 抗体节点
            
            # 抗原到抗体边（双向）
            ag2ab_mask = ag_mask[:, None] & ab_mask[None, :]
            ag2ab_edges = self._knn_with_mask(dist, ag2ab_mask, self.k_neighbors//2)
            
            # 抗体到抗原边（双向）
            ab2ag_edges = self._knn_with_mask(dist, ag2ab_mask.T, self.k_neighbors//2)
            
            # 转换全局索引（修正点）
            base = torch.nonzero(mask).min()  # 当前批次的起始索引
            batch_edges = torch.cat([
                intra_edges + base,
                inter_edges + base,
                ag2ab_edges + base,
                ab2ag_edges + base
            ], dim=1)
            
            edges.append(batch_edges)  # 将当前批次的边加入列表

        return torch.cat(edges, dim=1) if edges else torch.empty(2,0).long().to(X.device) 
    
    def _knn_with_mask(self, dist, mask, k):
        """在掩码区域内选择k近邻"""
        dist = dist.masked_fill(~mask, 1e9)
        valid_counts = mask.sum(dim=1)
        # print(f"[DEBUG] 有效节点数统计 | min: {valid_counts.min().item():<3} max: {valid_counts.max().item():<3} | 当前k值: {k}")
        actual_k = min(k, valid_counts.max().item())
        # if actual_k < k:
        #     print(f"[WARNING] 有效节点数不足！要求k={k}，实际可用k={actual_k}")
        topk = torch.topk(dist, k=actual_k, dim=1, largest=False)
        rows = torch.arange(dist.size(0), device=dist.device)[:,None].expand_as(topk.indices)
        return torch.stack([rows.flatten(), topk.indices.flatten()])

    def forward(self, X, H, A, AP, segment_ids, batch_id, B, L):
        """ 
        X: [N, C, 3], 坐标 (N=总节点数)
        H: [N, D], 节点特征
        batch_id: [B*L], 批次ID 
        """

        # 1. 原子特征与权重
        atom_feat = self.atom_proj(H)
        h = atom_feat[:, :self.hidden_dim]  # [N, hidden_dim]
        weight = atom_feat[:, self.hidden_dim:]  # [N, n_channel]
        channel_weights = torch.softmax(weight, dim = -1)  # [N, hidden_dim]
        channel_mask = torch.argmax(channel_weights, dim=-1)
        channel_weights = F.one_hot(channel_mask, num_classes=self.n_channel).float()
        
        # 2. 原子类型/位置嵌入（示例，需根据输入数据调整）
        # 假设输入数据包含原子类型A和位置AP（需从外部传入）
        # 这里仅作演示，实际需根据数据接口调整
        # A = self._get_atom_types(X)          # [N, C] 原子类型索引
        # AP = self._get_atom_positions(X)     # [N, C] 原子位置索引
        
        # 分通道嵌入（保持与dyMEAN一致的多头嵌入）
        atom_emb = (
            self.atom_type_embed(A) + 
            self.atom_pos_embed(AP)
        )

        # 3. 构建通道属性（仿dyMEAN）
        channel_attr = torch.cat([
            X.detach(),          # [N, C, 3]
            atom_emb             # [N, C, D/2]
        ], dim=-1)               # [N, C, 3 + D/2]
        
        # 4. 构建边
        edges = self._build_edges(X, segment_ids, batch_id)
        row_seg = segment_ids[edges[0]]
        col_seg = segment_ids[edges[1]]

        intra_ab_mask = (row_seg > 0) & (col_seg > 0)      # 抗体-抗体
        inter_ag_ab_mask = (row_seg == 0) ^ (col_seg == 0) # 抗原-抗体交叉边

        intra_edge_index = edges[:, intra_ab_mask]
        inter_edge_index = edges[:, inter_ag_ab_mask]
        # 5. 标准化坐标
        X_norm = self.normalizer.centering(X, batch_id)
        ctx_states = []
        # 6. GNN消息传递
        for ctx_layer, inter_layer in zip(self.ctx_layers, self.inter_layers):
            # 上下文路径（处理链内信息）
            h_ctx, X_norm = ctx_layer(
                h=h,
                x=X_norm,
                edge_index=intra_edge_index,  # 链内边
                channel_attr=channel_attr,
                channel_weights=channel_weights,
                radial_linear=self.radial_linear
            )
            
            # 交互路径（处理链间/抗原-抗体边）
            h_inter, X_norm = inter_layer(
                h=h,
                x=X_norm,
                edge_index=inter_edge_index,  # 链间边
                channel_attr=channel_attr,
                channel_weights=channel_weights,
                radial_linear=self.radial_linear
            )
            
            gate = self.fusion_gate(torch.cat([h_ctx, h_inter], dim=-1))
            h = gate * h_ctx + (1 - gate) * h_inter
            ctx_states.append(h)
        
        # 7. 恢复坐标
        X_out = self.normalizer.uncentering(X_norm, batch_id)
        h = h.view(B, L, -1)
        # print("DEBUG: GNNEncoder output shape:", h.shape)
        return h

class AM_E_GCL(nn.Module):
    """与dyMEAN完全兼容的AM_E_GCL实现"""
    def __init__(self, input_nf, hidden_nf, output_nf, n_channel, channel_nf, radial_nf, dropout=0.1):
        super().__init__()
        self.n_channel = n_channel
        self.radial_dim = radial_nf
        
        # 边模型（与原版参数一致）
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*input_nf + radial_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )
        
        # 节点模型（残差连接）
        self.node_mlp = nn.Sequential(
            nn.Linear(input_nf + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, output_nf)
        )
        
        # 坐标模型核心组件
        self.roller_pool = RollerPooling(n_channel)  # 使用移植的滚动池化
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, n_channel)
        )
        
        # 注意力机制
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid()
        ) if hidden_nf == input_nf else None
        
        self.norm = nn.LayerNorm(output_nf)

    def forward(self, h, x, edge_index, channel_attr, channel_weights, radial_linear):
        row, col = edge_index
        
        # 1. 调用移植的coord2radial
        radial, coord_diff = coord2radial(
            edge_index=(row, col),
            coord=x,
            attr=channel_attr,
            channel_weights=channel_weights,
            linear_map=radial_linear
        )  # radial: [E, D], coord_diff: [E, C, 3]
        
        # 2. 边特征计算（带通道注意力）
        edge_feat = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))
        if self.att_mlp is not None:
            edge_feat = edge_feat * self.att_mlp(edge_feat)
        
        # 3. 滚动池化实现多尺度特征
        channel_sum = (channel_weights != 0).sum(-1)[row]  # [E]
        pooled_feat = self.roller_pool(
            self.coord_mlp(edge_feat),  # [E, C]
            channel_sum
        ).repeat(1, 1, 3)  # [E, C, 1]
        # print("DEBUG: pooled_feat.squeeze(-1) shape:", pooled_feat.squeeze(-1).shape)
        # print("DEBUG: coord_diff shape:", coord_diff.shape)
        # 4. 等变坐标更新
        trans = coord_diff * pooled_feat.squeeze(-1)  # [E, C, 3]
        x = x + scatter_mean(trans, row, dim=0, dim_size=x.size(0))  # [N, C, 3]
        
        # 5. 节点更新（残差连接）
        agg = scatter_mean(edge_feat, col, dim=0, dim_size=h.size(0))
        h = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))
        
        return h, x

# class AM_E_GCL(nn.Module):
#     """ 完整实现与dyMEAN兼容的层 """
#     def __init__(self, input_nf, hidden_nf, output_nf, n_channel, channel_nf, radial_nf, dropout=0.1):
#         super().__init__()
#         self.input_nf = input_nf
#         self.output_nf = output_nf
#         self.radial_nf = radial_nf
#         self.n_channel = n_channel
#         # 边模型（与原版参数一致）
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2*input_nf + radial_nf, hidden_nf),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_nf, hidden_nf),
#             nn.SiLU()
#         )
        
#         # 节点模型
#         self.node_mlp = nn.Sequential(
#             nn.Linear(input_nf + hidden_nf, hidden_nf),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_nf, output_nf)
#         )
        
#         # 坐标模型（需使用dyMEAN的RollerPooling）
#         # self.roller_pool = RollerPooling(n_channel)
#         self.rot_emb = nn.Linear(9, hidden_nf)  # 处理3x3旋转矩阵
#         self.trans_emb = nn.Linear(3, hidden_nf)  # 处理平移向量

#         self.coord_mlp = nn.Sequential(
#             nn.Linear(hidden_nf, hidden_nf*n_channel),
#             nn.SiLU(),
#             nn.Linear(hidden_nf*n_channel, 3 * n_channel)
#         )
        
#         # 注意力机制（与原版一致）
#         self.att_mlp = nn.Sequential(
#             nn.Linear(hidden_nf, hidden_nf),
#             nn.SiLU(),
#             nn.Linear(hidden_nf, n_channel),  # 多通道注意力
#             nn.Softmax(dim=-1)
#             )

#         self.norm = nn.LayerNorm(output_nf)
    
#     def forward(self, h, x, edge_index, channel_attr, channel_weights, radial_linear):
#         row, col = edge_index
#         # 使用dyMEAN的coord2radial
#         R = construct_3d_basis(x[:,1], x[:,2], x[:,0])  # 从原子坐标构建局部坐标系
#         t = x[:,1]  # CA原子坐标为原点
#         # radial, coord_diff = coord2radial(
#         #     edge_index=(row, col),
#         #     coord=x,
#         #     attr=channel_attr,  # [N, C, 3+D/2]
#         #     channel_weights=channel_weights,
#         #     linear_map=radial_linear
#         # )
        
#         # 边特征计算
#         # edge_feat = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))
#         # edge_feat = edge_feat * self.att_mlp(edge_feat)
#         geom_feat = self.rot_emb(R.view(-1,9)) + self.trans_emb(t)
#         edge_attr = radial_linear(channel_attr)  # [E,5,256]
#         edge_attr = edge_attr.mean(dim=1)       # 沿通道维度平均 [E,256]
#         print("DEBUG: AM_E_GCL edge_attr shape:", radial_linear(channel_attr).shape)
#         print("DEBUG: AM_E_GCL geom_feat shape:", geom_feat[col].shape)
#         edge_attr = edge_attr + geom_feat[col]  # 与原版dyMEAN一致
        
#         # 修正坐标更新维度
#         coord_delta = self.coord_mlp(edge_attr).view(-1, self.n_channel, 3)  # [E,5,3]
#         # 节点更新
#         # agg = scatter_mean(edge_feat, col, dim=0, dim_size=h.size(0))
#         # h_new = self.node_mlp(torch.cat([h, agg], dim=-1)) + h  # 残差连接
        
#         # 坐标更新（等变）
#         # coord_delta = self.coord_mlp(edge_feat).view(-1, self.n_channel, 3)  # [E,5,3]
        
#         # # 修正聚合维度：
#         # # 将通道维度展开到节点维度，使用扩展后的col索引
#         # col_expanded = col.unsqueeze(1).expand(-1, self.n_channel).reshape(-1)
#         # coord_delta_flat = coord_delta.view(-1, 3)  # [E*5, 3]
#         col_expanded = col.unsqueeze(1).repeat(1, self.n_channel).view(-1)
#         coord_delta_flat = coord_delta.view(-1, 3)  # [E*5, 3]
#         # 散射求和时需要指定正确的输出维度（N*5）
#         x_flat = x.view(-1, 3)  # 将原始坐标展开为[N*5, 3]
#         x_flat_new = x_flat + scatter_add(
#             coord_delta_flat, 
#             col_expanded, 
#             dim=0, 
#             dim_size=x.size(0)  # 关键修改点
#         )
#         x_new = x_flat_new.view(x.shape)  # 恢复形状[N,5,3]
#         h_new = self.norm(h + self.node_mlp(torch.cat([h, edge_attr], dim=-1)))
#         return h_new, x_new

class EnhancedGNNEncoder(nn.Module):
    """整合dyMEAN的多通道几何编码与自适应边构造"""
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers):
        super().__init__()
        # 多尺度特征投影
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim*2),
            nn.SiLU(),
            nn.Linear(node_feat_dim*2, node_feat_dim)
        )
        # 多通道几何编码器
        self.geom_encoder = MultiChannelEncoder(
            node_dim=node_feat_dim,
            edge_dim=pair_feat_dim,
            n_layers=num_layers,
            n_channel=3  # 主链N, CA, C三个通道
        )
        
        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(node_feat_dim*2, node_feat_dim),
            nn.Sigmoid()
        )
        
        # 输出归一化
        self.layer_norm = nn.LayerNorm(node_feat_dim)

    def forward(self, R, t, res_feat, pair_feat, mask_res):
        """
        R: 旋转矩阵 [B, L, 3, 3]
        t: 平移向量 [B, L, 3]
        res_feat: ResidueEmbedding输出 [B, L, D]
        pair_feat: PairEmbedding输出 [B, L, L, E]
        """
        B, L = mask_res.shape
        device = mask_res.device
        
        # 1. 构建多通道坐标 (N, CA, C)
        bb_coords = torch.stack([
            t + R[..., 0],  # N原子
            t,              # CA原子
            t + R[..., 2]   # C原子
        ], dim=2)  # [B, L, 3, 3]
        
        # 2. 残基特征增强
        h_res = self.node_proj(res_feat)  # [B, L, D]
        
        # 3. 批次处理
        batch_id = torch.arange(B, device=device)[:, None].expand(B, L).flatten()  # [B*L]
        bb_coords_flat = bb_coords.view(B*L, 3, 3)  # [B*L, 3, 3]
        
        # 4. 多通道几何编码
        h_geom, _ = self.geom_encoder(
            h=h_res.view(B*L, -1),
            x=bb_coords_flat,
            edge_attr=pair_feat.view(B*L*L, -1),
            batch_id=batch_id
        )  # [B*L, D]
        
        # 5. 门控特征融合
        gate = self.fusion_gate(torch.cat([
            h_res.view(B*L, -1),
            h_geom
        ], dim=-1))  # [B*L, D]
        h_out = gate * h_geom + (1 - gate) * h_res.view(B*L, -1)
        
        # 6. 残基级聚合与归一化
        h_out = self.layer_norm(h_out.view(B, L, -1))
        return h_out * mask_res[..., None]

class MultiChannelEncoder(nn.Module):
    """自适应多通道几何编码器"""
    def __init__(self, node_dim, edge_dim, n_layers, n_channel=3):
        super().__init__()
        self.gcl_layers = nn.ModuleList([
            AM_E_GCL(
                input_nf=node_dim,
                output_nf=node_dim,
                hidden_nf=node_dim*2,
                n_channel=n_channel,
                channel_nf=32,
                radial_nf=128,
                edges_in_d=edge_dim
            ) for _ in range(n_layers)
        ])
        
        # 边构造系统
        self.edge_conv = GMEdgeConstructor(
            boa_idx=-1, boh_idx=-1, bol_idx=-1,
            atom_pos_pad_idx=-1, ag_seg_id=1
        )

    def forward(self, h, x, edge_attr, batch_id):
        # 构造层次化边
        ctx_edges, inter_edges = self.edge_conv.construct_edges(
            X=x[:,0],  # 使用Cα坐标构造边
            S=torch.zeros(len(x), device=x.device),  # 伪残基类型
            batch_id=batch_id,
            k_neighbors=30
        )
        
        # 多轮消息传递
        for layer in self.gcl_layers:
            h, x = layer(
                h=h,
                edge_index=torch.cat([ctx_edges, inter_edges], dim=1),
                coord=x,
                channel_attr=self._get_channel_attr(x),
                channel_weights=self._get_channel_weights(h, x),
                edge_attr=edge_attr
            )
        return h, x

    def _get_channel_attr(self, x):
        """生成通道属性（协方差矩阵特征）"""
        mean = x.mean(dim=1, keepdim=True)
        cov = torch.bmm((x - mean).transpose(1,2), x - mean) / x.size(1)
        return cov[..., :3]  # 取前三维特征

    def _get_channel_weights(self, h, x):
        """自适应通道权重"""
        return torch.softmax(
            nn.Linear(h.shape[-1] + 3, 3)(torch.cat([h, x.mean(dim=1)], dim=-1)),
            dim=-1
        )

def _knn_edges(X, AP, src_dst, atom_pos_pad_idx, k_neighbors, batch_info, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param AP: [N, n_channel], atom position with pad type need to be ignored
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    if given_dist is None:
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = AP[src_dst] == atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * BIGINT  # [Ef, n_channel, n_channel]
        del pos_pad  # release memory
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    else:
        dist = given_dist
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src
    return edges  # [2, E]

class EdgeConstructor:
    def __init__(self, boa_idx, boh_idx, bol_idx, atom_pos_pad_idx, ag_seg_id) -> None:
        self.boa_idx, self.boh_idx, self.bol_idx = boa_idx, boh_idx, bol_idx
        self.atom_pos_pad_idx = atom_pos_pad_idx
        self.ag_seg_id = ag_seg_id

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        outer_edges = _knn_edges(
            X, atom_pos, torch.stack([inter_all_row, inter_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return outer_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            self.not_global_edges,  # not global edges (also ensure the edges are in the same segment)
            self.row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos, segment_ids):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        ctx_edges, inter_edges = [], []

        # edges within chains
        inner_edges = self._construct_inner_edges(X, batch_id, k_neighbors, atom_pos)
        # edges between global nodes and normal/global nodes
        global_normal, global_global = self._construct_global_edges()
        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        # construct context edges
        ctx_edges = torch.cat([inner_edges, global_normal, global_global, seq_edges], dim=1)  # [2, E]

        # construct interaction edges
        inter_edges = self._construct_outer_edges(X, batch_id, k_neighbors, atom_pos)

        self._reset_buffer()
        return ctx_edges, inter_edges


class GMEdgeConstructor(EdgeConstructor):
    '''
    Edge constructor for graph matching (kNN internel edges and all bipartite edges)
    '''
    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: both in ag or ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag == col_is_ag, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = sequential_and(
            self.row_global, self.col_global, # self-loop has been deleted
            row_is_ag == col_is_ag)  # only inter-ag or inter-ab globals
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: one in ag and one in ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag != col_is_ag, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])  # [2, E]


class SeperatedCoordNormalizer(nn.Module):
    """分离式坐标标准化（抗原/抗体分别处理）"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        
    def centering(self, X, batch_id):
        """分批次中心化"""
        centers = scatter_mean(X[:,1], batch_id, dim=0)  # 使用CA原子计算质心
        return X - centers[batch_id].unsqueeze(1)
    
    def uncentering(self, X, batch_id, centers=None):
        """恢复原始坐标系"""
        if centers is not None:
            return X + centers[batch_id].unsqueeze(1)
        return X
    