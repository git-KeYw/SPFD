import torch
import torch.nn as nn
from detrex.layers import MultiScaleDeformableAttention, FFN

class TriDeformAdapter6Layer(nn.Module):

    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 ffn_dim=1024,
                 num_feature_levels=4,
                 num_layers=6,
                 dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_r": MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_levels=num_feature_levels,
                    batch_first=True,
                    dropout=dropout,
                ),
                "cross_e": MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_levels=num_feature_levels,
                    batch_first=True,
                    dropout=dropout,
                ),
                "ffn": FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=ffn_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=dropout,
                ),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
            })
            for _ in range(num_layers)
        ])

        self.post_norm = nn.LayerNorm(embed_dim)

        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_r = nn.Linear(embed_dim, embed_dim)
        self.proj_e = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.proj_r.weight)
        nn.init.xavier_uniform_(self.proj_e.weight)
    # --- flatten 私有模态特征 ---
    def _flatten_feats(self, feats, spatial_shapes):
        outs = []
        for (h, w), f in zip(spatial_shapes.tolist(), feats):
            outs.append(f.flatten(2).transpose(1, 2))  # [B,HW,C]
        return torch.cat(outs, dim=1)

    # --- forward ---
    def forward(
            self,
            query,  # feat_flatten
            key=None,
            value=None,
            query_pos=None,  # lvl_pos_embed_flatten
            query_key_padding_mask=None,  # mask_flatten
            spatial_shapes=None,
            reference_points=None,
            level_start_index=None,
            valid_ratios=None,
            private_feats_r=None,
            private_feats_e=None,
            **kwargs,
    ):

        priv_r = self._flatten_feats(private_feats_r, spatial_shapes)
        priv_e = self._flatten_feats(private_feats_e, spatial_shapes)

        x = query + (query_pos if query_pos is not None else 0)
        memory_per_layer = []

        for layer in self.layers:
            # Cross-Attn from private features
            cross_r = layer["cross_r"](
                query=x,
                value=priv_r,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=query_key_padding_mask,
            )
            cross_e = layer["cross_e"](
                query=x,
                value=priv_e,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=query_key_padding_mask,
            )

            cross_mean = 0.5 * (cross_r + cross_e)
            gate = torch.sigmoid(self.gate_proj(cross_mean))
            mix_r = self.proj_r(cross_r)
            mix_e = self.proj_e(cross_e)

            x = layer["norm1"](gate * mix_r + (1 - gate) * mix_e)
            x = layer["norm2"](layer["ffn"](x))
            memory_per_layer.append(x)

        memory_new = self.post_norm(x)
        return memory_new, memory_per_layer