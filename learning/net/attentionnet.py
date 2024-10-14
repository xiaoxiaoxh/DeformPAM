import torch
import torch.nn as nn
from learning.components.mlp import MLP_V2
from learning.net.layers import (
    FFWRelativeSelfCrossAttentionModule
)
from learning.net.position_encodings import (
    RotaryPositionEncoding,
    RotaryPositionEncoding3D,
    RotaryPositionEncoding6D,
    SinusoidalPosEmb
)
from learning.net.multihead_relative_attention import MultiheadCustomAttention


class PairFFWRelativeSelfCrossAttentionModule(FFWRelativeSelfCrossAttentionModule):
    def __init__(self, embedding_dim, num_attn_heads,
                 num_self_attn_layers, num_cross_attn_layers, num_gripper_points=2,
                 use_adaln=True,):
        super().__init__(embedding_dim=embedding_dim,
                            num_attn_heads=num_attn_heads,
                            num_self_attn_layers=num_self_attn_layers,
                            num_cross_attn_layers=num_cross_attn_layers,
                            use_adaln=use_adaln)
        self.num_gripper_points = num_gripper_points
    
    @staticmethod
    def _seq_to_pair(query, query_pos, num_gripper_points):
        K, B, C = query.shape
        query = query.reshape(K//num_gripper_points, num_gripper_points, B, C)
        query = query.permute(1, 2, 0, 3)
        query = query.reshape(num_gripper_points, B * K//num_gripper_points, C)
        if query_pos is not None:
            query_pos = query_pos.reshape(B * K//num_gripper_points, num_gripper_points, C, -1)
        return query, query_pos
    
    @staticmethod
    def _pair_to_seq(query, query_pos, K, B, C, num_gripper_points):
        query = query.reshape(num_gripper_points, B, K//2, C)
        query = query.permute(2, 0, 1, 3)
        query = query.reshape(K, B, C)
        if query_pos is not None:
            query_pos = query_pos.reshape(B, K, C, -1)
        return query, query_pos

    def forward(self, query, context, diff_ts=None,
                query_pos=None, context_pos=None):
        K, B, C = query.shape
        diff_ts_expanded = diff_ts.unsqueeze(1).expand(-1, K//self.num_gripper_points, -1).reshape(B*K//self.num_gripper_points, C)
        output = []
        for i in range(self.num_layers):
            # Cross attend to the context first
            if self.cross_attn_layers[i] is not None:
                if context_pos is None:
                    cur_query_pos = None
                else:
                    cur_query_pos = query_pos
                query = self.cross_attn_layers[i](
                    query, context, diff_ts, cur_query_pos, context_pos
                )
            # Self attend next
            # change shape to avoid information leakage
            query, query_pos = self._seq_to_pair(query, query_pos, num_gripper_points=self.num_gripper_points)
            query = self.self_attn_layers[i](
                query, query, diff_ts_expanded, query_pos, query_pos
            )
            query, query_pos = self._pair_to_seq(query, query_pos, K, B, C, num_gripper_points=self.num_gripper_points)
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class AttentionNet(nn.Module):
    def __init__(self, 
                 data_dim: int,
                 feature_dim: int,
                 num_layers: int,
                 num_heads: int,
                 num_gripper_points:int = 2,
                 use_positional_encoding: bool = False,
                 action_input_mlp_nn_channels: tuple = (120, 240),
                 action_output_mlp_nn_channels: tuple = (240, 120),
                 enable_extra_outputting_dims=False
                 ):
        super().__init__()

        self.feature_dim = feature_dim
        self.data_dim = data_dim
        self.enable_extra_outputting_dims = enable_extra_outputting_dims

        self.pos_emb = nn.Sequential(
            SinusoidalPosEmb(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # TODO: more flexible
        if self.enable_extra_outputting_dims:
            self.relative_pe_layer = RotaryPositionEncoding3D(feature_dim)
        else:
            if data_dim == 3:
                self.relative_pe_layer = RotaryPositionEncoding3D(feature_dim)
            elif data_dim == 6:
                self.relative_pe_layer = RotaryPositionEncoding6D(feature_dim)
            else:
                raise NotImplementedError
        
        # TODO: more flexible
        if self.enable_extra_outputting_dims:
            action_input_mlp_nn_channels = (3, *action_input_mlp_nn_channels)
        else:
            action_input_mlp_nn_channels = (data_dim, *action_input_mlp_nn_channels)
        action_output_mlp_nn_channels = (*action_output_mlp_nn_channels, data_dim)
        self.action_input_mlp = MLP_V2(action_input_mlp_nn_channels, transpose_input=True)
        self.action_output_mlp = MLP_V2(action_output_mlp_nn_channels, transpose_input=True)

        self.attn_layers = PairFFWRelativeSelfCrossAttentionModule(
            embedding_dim=feature_dim,
            num_attn_heads=num_heads,
            num_self_attn_layers=num_layers,
            num_cross_attn_layers=num_layers,
            num_gripper_points=num_gripper_points,
            use_adaln=True
        )
        self.use_positional_encoding = use_positional_encoding

    def forward(self,
                action: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                context_pos: torch.Tensor) -> torch.Tensor:
        # only for inference
        if len(timestep.shape) == 0:
            timestep = torch.Tensor([timestep]).to(context.device)
        
        context = context.permute(1, 0, 2)
        context_pos = self.relative_pe_layer(context_pos)

        action_feature = self.action_input_mlp(action)
        action_feature = action_feature.permute(1, 0, 2)
        action_pos = self.relative_pe_layer(action)

        if self.use_positional_encoding:
            pos = torch.zeros(action_feature.shape[0]).to(action.device)
            pos[::2] = 0
            pos[1::2] = 1
            pos_embedding = self.pos_emb(pos).unsqueeze(1)
            action_feature += pos_embedding
        time_embedding = self.time_emb(timestep)

        attention_output = self.attn_layers(
            query = action_feature,
            context = context,
            query_pos = action_pos,
            context_pos = context_pos,
            diff_ts = time_embedding
        )[-1]

        action_pred = self.action_output_mlp(attention_output).permute(1, 0, 2)
        return action_pred
        
    def get_optim_groups(self, weight_decay: float=1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.MultiheadAttention, MultiheadCustomAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups