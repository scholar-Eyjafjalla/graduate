# File: diffab/modules/adapters/dymean_adapter.py

import torch
import torch.nn as nn
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.modules.common.geometry import construct_3d_basis, global_to_local

class DiffabToDyMeanAdapter(nn.Module):
    """将Diffab数据转换为dyMEAN兼容格式的完整适配器"""
    def __init__(self, cfg):
        super().__init__()
        self.max_num_atoms = 4  # 对应N/CA/C/O 4个原子
        # 原始Diffab组件
        self.res_feat_dim = 128  
        # print(f"[Debug] Adapter.res_feat_dim: {self.res_feat_dim}")  # 预期输出 128
        # self.atom_type_embed = nn.Embedding(
        #     num_embeddings=4,  # 原子类型数 (N,CA,C,O,CB)
        #     embedding_dim=cfg.get('atom_type_embed_dim', 32)
        # )
        self.residue_embed = ResidueEmbedding(
            feat_dim=self.res_feat_dim,
            max_num_atoms=self.max_num_atoms,  # 明确指定原子数
            max_aa_types=cfg.get('max_aa_types', 22)  # 默认22类氨基酸
        )
        # dyMEAN风格的特征组件
        self.aa_vocab_size = 20  # 标准氨基酸类型数
        self.atom_type_size = 4   # N/CA/C/O
        
        # 链类型编码 (抗原:0, 重链H:1, 轻链L:2)
        self.chain_type_embed = nn.Embedding(3, cfg['res_feat_dim'])
        
        # 原子类型编码 (N/CA/C/O)
        # self.atom_type_embed = nn.Embedding(self.atom_type_size+1, 32, padding_idx=0)
        
        # 原子位置编码 (主链:1-3, O:4)
        # self.atom_pos_embed = nn.Embedding(5, 32)
        
        # 残基位置编码 (序列位置)
        self.res_pos_embed = nn.Embedding(512, 64)  # 最大长度512

        self.gnn_feat_mixer = nn.Sequential(
            nn.Linear(self.res_feat_dim + 64, self.res_feat_dim), 
            nn.ReLU(),
        )

    def _map_aa_type(self, aa):
        """将Diffab的氨基酸类型映射到dyMEAN的词汇表"""
        # Diffab使用0-19表示标准氨基酸，20为UNK
        return torch.clamp(aa, max=self.aa_vocab_size-1)

    def _get_segment_ids(self, batch):
        """生成dyMEAN风格的segment_id (抗原:0, 抗体H:1, 抗体L:2)"""
        segment = torch.zeros_like(batch['fragment_type'])
        
        # 抗原: fragment_type == 3
        segment = torch.where(batch['fragment_type'] == 3, 0, segment)
        
        # 重链H: fragment_type == 1
        segment = torch.where(batch['fragment_type'] == 1, 1, segment)
        
        # 轻链L: fragment_type == 2
        segment = torch.where(batch['fragment_type'] == 2, 2, segment)
        
        return segment

    def _get_atom_features(self, pos_atoms, mask_atoms):
        """生成原子级特征"""
        B, L = pos_atoms.shape[:2]
        
        # 原子类型映射 (N:0, CA:1, C:2, O:3)
        atom_types = torch.arange(1, 5, device=pos_atoms.device
                                ).expand(B, L, 4).clone()
        
        # 原子位置编码 (N:1, CA:2, C:3, O:4)
        atom_pos = torch.tensor([1,2,3,4], device=pos_atoms.device
                               ).expand(B, L, 4).clone()
        
        # 应用原子掩码 (无效原子设为0)
        atom_types = torch.where(mask_atoms, atom_types, 0)  # 使用0作为padding索引
        atom_pos = torch.where(mask_atoms, atom_pos, 0)      # 使用0作为padding索引
        
        return atom_types, atom_pos

    def forward(self, batch):
        bb_atom_indices = [
            BBHeavyAtom.N, 
            BBHeavyAtom.CA, 
            BBHeavyAtom.C, 
            BBHeavyAtom.O
        ]
        # 基础维度
        B, L = batch['aa'].shape
        device = batch['aa'].device
        
        # 原子坐标处理 (N/CA/C/O)
        pos_atoms = batch['pos_heavyatom'][..., bb_atom_indices, :]  # [B, L, 4, 3]
        mask_atoms = batch['mask_heavyatom'][..., bb_atom_indices]   # [B, L, 4]
        
        # ----------------------------
        # 生成dyMEAN风格的特征
        # ----------------------------
        
        # 1. 残基特征
        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
            fragment_type=batch['fragment_type']
        )  # [B, L, D_res]
        
        # 2. 链类型特征
        chain_type = self._get_segment_ids(batch)
        chain_feat = self.chain_type_embed(chain_type)  # [B, L, D_res]
        
        # 3. 原子类型特征
        atom_types, atom_pos = self._get_atom_features(pos_atoms, mask_atoms)
        # atom_type_feat = self.atom_type_embed(atom_types)  # [B, L, 4, 32]
        # atom_pos_feat = self.atom_pos_embed(atom_pos)      # [B, L, 4, 32]
        # atom_feat = atom_type_feat + atom_pos_feat        # [B, L, 4, 32]
        
        # 4. 残基位置特征
        res_pos = torch.clamp(batch['res_nb'], max=511)  # 限制最大位置
        res_pos_feat = self.res_pos_embed(res_pos)       # [B, L, 64]
        
        # ----------------------------
        # 特征拼接与聚合
        # ----------------------------
        
        # 节点级特征 (残基+链类型)
        node_feat = torch.cat([
            res_feat + chain_feat,  # [B, L, D_res]
            res_pos_feat            # [B, L, 64]
        ], dim=-1)  # [B, L, D_res+64]
        
        # 原子坐标转换到局部坐标系
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA],
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N]
        )  # [B, L, 3, 3]
        t = pos_atoms[:, :, BBHeavyAtom.CA]  # [B, L, 3]
        
        # local_coords = global_to_local(
        #     R, t, pos_atoms
        # )  # [B, L, 4, 3]
        
        node_feat = self.gnn_feat_mixer(node_feat)

        # print(f"[Debug] Adapter.H: {node_feat.shape}")

        return {
            # dyMEAN需要的核心特征
            'X': pos_atoms,          # 全局坐标 [B, L, 5, 3]
            # 'X_local': local_coords, # 局部坐标 [B, L, 5, 3]
            'H': node_feat,          # 节点特征 [B, L, D_node]
            'A': atom_types,         # 原子类型 [B, L, 5]
            'AP': atom_pos,          # 原子位置 [B, L, 5]
            'segment_ids': self._get_segment_ids(batch),  # [B, L]
            'batch_id': self._get_batch_id(B, L, device), # [B*L]
            'mask': mask_atoms.any(dim=-1)  # 残基掩码 [B, L]
        }

    def _get_batch_id(self, B, L, device):
        """生成批处理ID [B*L,]"""
        return torch.repeat_interleave(
            torch.arange(B, device=device),
            repeats=L
        )