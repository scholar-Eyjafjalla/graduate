import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.adapters.dymean_adapter import DiffabToDyMeanAdapter
from diffab.modules.encoders.ga import NewGNNEncoder
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.adapter = DiffabToDyMeanAdapter(cfg)
        self.gnn_encoder = NewGNNEncoder(
            node_feat_dim=self.adapter.res_feat_dim,
            hidden_dim=128,  # 强制指定
            num_layers=cfg.get('num_layers', 4),
            radial_dim=cfg.get('radial_dim', 256),
            k_neighbors=cfg.get('k_neighbors', 30),
            dropout=cfg.get('dropout', 0.1)
        )
        
        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        # self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            # cfg.pair_feat_dim,
            type_embed_weights=self.residue_embed.type_embed.weight.data,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # assert dymean_data['X'].shape[:2] == (B, L), "适配器输出维度不匹配"
        # assert dymean_data['batch_id'].shape == (B * L,), "batch_id形状错误"
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        # context_mask = torch.logical_and(
        #     batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
        #     ~batch['generate_flag']     # Context means ``not generated''
        # )

        # structure_mask = context_mask if remove_structure else None
        # sequence_mask = context_mask if remove_sequence else None

        # res_feat = self.residue_embed(
        #     aa = batch['aa'],
        #     res_nb = batch['res_nb'],
        #     chain_nb = batch['chain_nb'],
        #     pos_atoms = batch['pos_heavyatom'],
        #     mask_atoms = batch['mask_heavyatom'],
        #     fragment_type = batch['fragment_type'],
        #     structure_mask = structure_mask,
        #     sequence_mask = sequence_mask,
        # )

        # pair_feat = self.pair_embed(
        #     aa = batch['aa'],
        #     res_nb = batch['res_nb'],
        #     chain_nb = batch['chain_nb'],
        #     pos_atoms = batch['pos_heavyatom'],
        #     mask_atoms = batch['mask_heavyatom'],
        #     structure_mask = structure_mask,
        #     sequence_mask = sequence_mask,
        # )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        dymean_data = self.adapter(batch)
        B, L = dymean_data['X'].shape[:2]
        X = dymean_data['X'].view(B*L, 4, 3)
        H = dymean_data['H'].view(B*L, -1)
        A = dymean_data['A'].view(B*L, 4)
        AP = dymean_data['AP'].view(B*L, 4)
        segment_ids = dymean_data['segment_ids'].view(B*L)
        batch_id = dymean_data['batch_id'].view(B*L)

        gnn_feat = self.gnn_encoder(
            X=X,
            H=H,
            A=A,
            AP=AP,
            segment_ids=segment_ids,
            batch_id=batch_id,
            B=B,
            L=L
        )

        #fused_feat = res_feat + gnn_feat
        return gnn_feat, R, p
        return gnn_feat, pair_feat, R, p
        # return res_feat, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        #########################################################################
        gnn_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            # v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            batch, gnn_feat,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj
