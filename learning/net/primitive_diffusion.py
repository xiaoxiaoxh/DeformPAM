import numpy as np
import os
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
import time
import bson
from typing import Tuple, Optional
from loguru import logger
from learning.components.mlp import MLP_V2
import pytorch_lightning as pl
from learning.net.resunet import SparseResUNet
from learning.net.pointnet import MiniPointNetfeat
from learning.net.transformer import Transformer
from common.datamodels import PredictionMessage, ActionTypeDef, GeneralObjectState, GarmentSmoothingStyle, ObjectState, ActionIteratorMessage
import MinkowskiEngine as ME
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from learning.net.attentionnet import AttentionNet

from sklearn.metrics import roc_auc_score

import py_cli_interaction

class CustomScheduler:
    """
    add useful tools for DDIMScheduler
    """

    def get_noise(
        self,
        original_samples: torch.FloatTensor,
        noisy_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = (noisy_samples - sqrt_alpha_prod * original_samples) / sqrt_one_minus_alpha_prod
        return noise
    
    def get_minsnr_weights(
        self,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=timesteps.device)

        SNR = alphas_cumprod / (1 - alphas_cumprod)
        FIVE = torch.ones_like(SNR) * 5.0
        ONE = torch.ones_like(SNR)
        if self.config.prediction_type == "sample":
            weights = torch.minimum(SNR, FIVE)
        elif self.config.prediction_type == "epsilon":
            weights = torch.minimum(FIVE / SNR, ONE)
        else:
            raise NotImplementedError
        
        weights = weights[timesteps]
        weights = weights.flatten()
        
        return weights
    
class CustomDDIMScheduler(DDIMScheduler, CustomScheduler):
    pass

class CustomDDPMScheduler(DDPMScheduler, CustomScheduler):
    pass
    
class DiffusionHead(pl.LightningModule):
    def __init__(
            self,
            weight_decay: float,
            # diffusion params
            scheduler_type: str = 'ddim',
            ddim_eta: float = 0.0,
            num_training_steps: int = 100,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = 'squaredcos_cap_v2',
            clip_sample: bool = True,
            set_alpha_to_one: bool = True,
            steps_offset: int = 0,
            prediction_type: str = 'sample',
            num_inference_steps: Optional[int] = 10,
            num_of_grasp_points: int = 8,
            num_gripper_points: int = 2,
            # data params
            data_format: str = 'nocs',
            adaptive_xyz: bool = False,
            match_idx: Optional[Tuple] = None,
            # model params
            feature_dim: int = 240,
            num_diffusion_net_layers: int = 2,
            num_diffusion_net_heads: int = 4,
            action_input_mlp_nn_channels: Tuple[int, int] = (120, 240),
            action_output_mlp_nn_channels: Tuple[int, int] = (240, 120),
            use_positional_encoding_in_attention_net: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.weight_decay = weight_decay
        self.data_format = data_format
        self.feature_dim = feature_dim
        self.num_of_grasp_points = num_of_grasp_points
        self.num_gripper_points = num_gripper_points
        self.kwargs = kwargs
        if data_format == 'nocs' or data_format == 'xyz':
            self.data_dim = 3
            self.adaptive_xyz = False
        elif data_format == 'xyznocs':
            self.data_dim = 6
            self.adaptive_xyz = adaptive_xyz
        else:
            raise NotImplementedError
        self.match_idx = match_idx

        assert action_input_mlp_nn_channels[-1] == feature_dim and action_output_mlp_nn_channels[0] == feature_dim, \
            "The channel of action MLP should match the feature dim"
        self.model = AttentionNet(
            data_dim=self.data_dim,
            feature_dim=feature_dim,
            num_layers=num_diffusion_net_layers,
            num_heads=num_diffusion_net_heads,
            num_gripper_points=num_gripper_points,
            use_positional_encoding=use_positional_encoding_in_attention_net,
            action_input_mlp_nn_channels=action_input_mlp_nn_channels,
            action_output_mlp_nn_channels=action_output_mlp_nn_channels,
            )
        
        assert scheduler_type in ['ddim', 'ddpm'], "Only support DDIM and DDPM scheduler now"
        if prediction_type == 'epsilon':
            assert data_format == 'xyz', "Only support xyz data format for epsilon prediction type"
            logger.warning("Only support xyz data format for epsilon prediction type")
            logger.warning("The variety error has no meaning for epsilon prediction type")
        if scheduler_type == 'ddim':
            self.noise_scheduler = CustomDDIMScheduler(
                num_train_timesteps=num_training_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                set_alpha_to_one=set_alpha_to_one,
                steps_offset=steps_offset,
                prediction_type=prediction_type,
            )
        elif scheduler_type == 'ddpm':
            self.noise_scheduler = CustomDDPMScheduler(
                num_train_timesteps=num_training_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        self.scheduler_type = scheduler_type
        self.ddim_eta = ddim_eta

        if num_inference_steps is None:
            num_inference_steps = num_training_steps
        self.num_inference_steps = num_inference_steps
        self.num_training_steps = num_training_steps

    def _get_action_xyz_nocs(self,
                             action: torch.Tensor,
                             pc_xyz: torch.Tensor,
                             pred_pc_nocs: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        if self.data_format == 'nocs' or self.adaptive_xyz:
            pred_pc_nocs_expanded = pred_pc_nocs.unsqueeze(1).expand(-1, self.num_of_grasp_points, -1, -1)
            action_expanded = action.unsqueeze(2).expand(-1, -1, pred_pc_nocs.shape[1], -1)
            if self.data_format == 'nocs':
                action_nocs = action
                action_nocs_expanded = action_expanded
            elif self.data_format == 'xyznocs':
                action_nocs = action[..., 3:]
                action_nocs_expanded = action_expanded[..., 3:]
            distance = torch.norm(pred_pc_nocs_expanded - action_nocs_expanded, dim=-1)
            index = torch.argmin(distance, dim=2)
            action_xyz = pc_xyz.gather(1, index.unsqueeze(-1).expand(-1, -1, 3))
        else:
            action_xyz = action[..., :3]
            if self.data_format == 'xyz':
                pc_xyz_expanded = pc_xyz.unsqueeze(1).expand(-1, self.num_of_grasp_points, -1, -1)
                action_xyz_expanded = action_xyz.unsqueeze(2).expand(-1, -1, pc_xyz.shape[1], -1)
                distance = torch.norm(pc_xyz_expanded - action_xyz_expanded, dim=-1)
                index = torch.argmin(distance, dim=2)
                action_nocs = pred_pc_nocs.gather(1, index.unsqueeze(-1).expand(-1, -1, 3))
            elif self.data_format == 'xyznocs':
                action_nocs = action[..., 3:]
            
        return action_xyz, action_nocs
    
    def _get_data(self, data_xyz: torch.Tensor, data_nocs: torch.Tensor) -> torch.tensor:
        if self.data_format == 'nocs':
            data = data_nocs
        elif self.data_format == 'xyz':
            data = data_xyz
        elif self.data_format == 'xyznocs':
            data = torch.cat([data_xyz, data_nocs], dim=-1)
        else:
            raise NotImplementedError   

        return data  
    
    def _get_matched_action_gt(self,
                               action_noisy: torch.Tensor,
                               multiple_action_xyz_gt: torch.Tensor,
                               multiple_action_nocs_gt: torch.Tensor,
                               timesteps: torch.Tensor) -> torch.tensor:
        
        B, N, _, D1 = multiple_action_xyz_gt.shape
        multiple_action_xyz_gt = multiple_action_xyz_gt.reshape(B*N, self.num_gripper_points, D1)
        _, _, _, D2 = multiple_action_nocs_gt.shape
        multiple_action_nocs_gt = multiple_action_nocs_gt.reshape(B*N, self.num_gripper_points, D2)
        multiple_action_gt = self._get_data(multiple_action_xyz_gt, multiple_action_nocs_gt)
        
        M = self.num_of_grasp_points // self.num_gripper_points
        action_noisy = action_noisy.reshape(B, M, self.num_gripper_points, -1)
        multiple_action_gt = multiple_action_gt.reshape(B, N, self.num_gripper_points, -1)
        B, N, _, D = multiple_action_gt.shape
        sym_multiple_action_gt = multiple_action_gt.clone()
        sym_multiple_action_gt[:, :, 0, :] = multiple_action_gt[:, :, 1, :]
        sym_multiple_action_gt[:, :, 1, :] = multiple_action_gt[:, :, 0, :]
        
        action_noisy_expanded = action_noisy.unsqueeze(2).expand(-1, -1, N, -1, -1)
        multiple_action_gt_expanded = multiple_action_gt.unsqueeze(1).expand(-1, M, -1, -1, -1)
        sym_multiple_action_gt_expanded = sym_multiple_action_gt.unsqueeze(1).expand(-1, M, -1, -1, -1)
        
        # choose the best match according to the distance from the noisy action
        metric = torch.nn.MSELoss(reduction='none')
        # only consider the match range
        if self.match_idx is None:
            self.match_idx = np.arange(D)
        else:
            match_idx = torch.tensor(self.match_idx).to(torch.long)
        distance = torch.minimum(
            metric(action_noisy_expanded[..., match_idx], multiple_action_gt_expanded[..., match_idx]).mean(dim=(3, 4)),
            metric(action_noisy_expanded[..., match_idx], sym_multiple_action_gt_expanded[..., match_idx]).mean(dim=(3, 4)))
        index = torch.argmin(distance, dim=2)
        
        if self.noise_scheduler.config.prediction_type == 'sample':
            action_gt = multiple_action_gt_expanded.gather(2, index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2, D))
        else:
            noise_expanded = self.noise_scheduler.get_noise(multiple_action_gt_expanded, action_noisy_expanded, timesteps)
            action_gt = noise_expanded.gather(2, index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2, D))
        action_gt = action_gt.reshape(B, M*2, D)
        return action_gt
    
    def get_minsnr_weights(self, timesteps: torch.IntTensor) -> torch.FloatTensor:
        return self.noise_scheduler.get_minsnr_weights(timesteps)

    def forward(self, 
                action: torch.Tensor, 
                timestep: torch.Tensor, 
                context: torch.Tensor, 
                context_pos: torch.Tensor) -> torch.tensor:
        return self.model(action, timestep, context, context_pos)
    
    def diffuse_denoise(self, 
                        action_xyz_gt: torch.Tensor,
                        action_nocs_gt: torch.Tensor,
                        context: torch.Tensor,
                        pc_xyz: torch.Tensor,
                        pc_nocs: torch.Tensor,
                        use_matched_action_gt: bool = True,
                        multiple_action_xyz_gt: torch.Tensor = None,
                        multiple_action_nocs_gt: torch.Tensor = None,
                        timesteps: torch.Tensor = None,
                        noise: torch.Tensor = None,
                        replicate_action: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        action_gt = self._get_data(action_xyz_gt, action_nocs_gt)
        context_pos = self._get_data(pc_xyz, pc_nocs)
        
        B, D = action_gt.shape[0], action_gt.shape[-1]
        device = action_gt.device
        noise_scheduler = self.noise_scheduler

        if replicate_action:
            action_gt = action_gt.unsqueeze(1).expand(-1, self.num_of_grasp_points // self.num_gripper_points, -1, -1).reshape(B, -1, D)
        if noise is None:
            noise = torch.randn_like(action_gt)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device
            ).long()
        action_noisy = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        action_pred = self.forward(action_noisy, timesteps, context, context_pos)

        action_xyz_pred, action_nocs_pred = self._get_action_xyz_nocs(action_pred, pc_xyz, pc_nocs)
        
        if use_matched_action_gt and multiple_action_xyz_gt is not None and multiple_action_nocs_gt is not None:
            action_gt = self._get_matched_action_gt(action_noisy, multiple_action_xyz_gt, multiple_action_nocs_gt, timesteps)
        else:
            if self.noise_scheduler.config.prediction_type == 'epsilon':
                action_gt = noise
        
        return action_gt, action_pred, action_xyz_pred, action_nocs_pred, timesteps, noise
        
    def conditional_sample(self,
                           context: torch.Tensor,
                           pc_xyz: torch.Tensor,
                           pred_pc_nocs: torch.Tensor,
                           generator=None,
                           **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        scheduler = self.noise_scheduler
        action = torch.randn(
            context.shape[0], self.num_of_grasp_points, self.data_dim
        ).to(context.device)
        action_xyz_list = []

        action_xyz, action_nocs = self._get_action_xyz_nocs(action, pc_xyz, pred_pc_nocs)
        if self.adaptive_xyz:
            action[..., :3] = action_xyz
        action_xyz_list.append(action_xyz)

        context_pos = self._get_data(pc_xyz, pred_pc_nocs)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.forward(action, t, context, context_pos)
            if self.scheduler_type == 'ddim':
                action = scheduler.step(
                    model_output, t, action, eta=self.ddim_eta, generator=generator, **kwargs
                ).prev_sample
            else:
                action = scheduler.step(
                    model_output, t, action, generator=generator, **kwargs
                ).prev_sample
            action_xyz, action_nocs = self._get_action_xyz_nocs(action, pc_xyz, pred_pc_nocs)
            if self.adaptive_xyz:
                action[..., :3] = action_xyz
            action_xyz_list.append(action_xyz)

        return action_xyz, action_nocs, action_xyz_list
        
class StateHead(pl.LightningModule):
    """concat nocs feature, global feature, dense feature as nocs feature input"""
    def __init__(self,
                 global_nn_channels: tuple = (128, 256, 1024),
                 cls_base_nn_channels: tuple = (1024, 256, 128),
                 pointnet_channels: tuple = (3, 64, 128, 512),
                 grasp_nocs_feat_nn_channels: tuple = (512 + 64 + 1024 + 128, 512, 256),
                 nocs_nn_channels: tuple = (128, 256, 128, 3),
                 offset_nn_channels: tuple = (128, 256, 128, 1),  # only predict (x, y) coordinate
                 att_nn_channels: tuple = (128, 256, 128, 1),
                 num_smoothing_style: int = 4,  # short: (down, up, left, right)
                 num_keypoints: int = 4, # (left shoulder, right shoulder, left waist, right waist)
                 min_gt_nocs_ratio: float = 0.2,
                 gt_nocs_ratio_decay_factor: float = 0.98,  # for 100 epoch setting
                 num_pred_fling_candidates: int = 4,  # number of possible fling candidates
                 use_xyz_variety_loss: bool = False,
                 use_gt_nocs_pred_for_distance_weight: bool = False,
                 nocs_distance_weight_alpha: float = 30.0,
                 use_nocs_for_dense_feat: bool = True,
                 detach_for_classifier: bool = False,
                 detach_for_detector: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_action_type = num_smoothing_style
        self.num_pred_fling_candidates = num_pred_fling_candidates
        self.num_keypoints = num_keypoints

        self.nocs_pointnet = MiniPointNetfeat(nn_channels=pointnet_channels)
        self.grasp_nocs_feat_mlp = MLP_V2(grasp_nocs_feat_nn_channels, transpose_input=True)
        
        self.offset_mlp_list = nn.ModuleList([
            MLP_V2(offset_nn_channels, transpose_input=True) for _ in range(num_keypoints)
        ])
        self.att_mlp_list = nn.ModuleList([
            MLP_V2(att_nn_channels, transpose_input=True) for _ in range(num_keypoints)
        ])

        self.global_mlp = MLP_V2(global_nn_channels, transpose_input=True)
        self.nocs_mlp = MLP_V2(nocs_nn_channels, transpose_input=True)
        self.smoothed_cls_mlp = MLP_V2(cls_base_nn_channels + (1,), transpose_input=True)
        self.smoothing_style_mlp = MLP_V2(cls_base_nn_channels + (num_smoothing_style,), transpose_input=True)

        self.gt_nocs_ratio = 1.0
        self.min_gt_nocs_ratio = min_gt_nocs_ratio
        self.gt_nocs_ratio_decay_factor = gt_nocs_ratio_decay_factor
        
        self.use_xyz_variety_loss = use_xyz_variety_loss
        self.use_gt_nocs_pred_for_distance_weight = use_gt_nocs_pred_for_distance_weight
        self.nocs_distance_weight_alpha = nocs_distance_weight_alpha
        
        self.use_nocs_for_dense_feat = use_nocs_for_dense_feat

        self.detach_for_classifier = detach_for_classifier
        self.detach_for_detector = detach_for_detector

    def forward(self, pc_xyz: torch.Tensor, dense_feat: torch.Tensor, gt_pc_nocs: torch.Tensor = None):
        dense_feat_extra = self.global_mlp(dense_feat)  # (B, N, C')
        global_feat = torch.max(dense_feat_extra, dim=1)[0]  # (B, C)
        if self.detach_for_classifier:
            smoothed_logits = self.smoothed_cls_mlp(global_feat.detach()) # stop gradient
            smoothing_style_logits = self.smoothing_style_mlp(global_feat.detach()) # stop gradient
        else:
            smoothed_logits = self.smoothed_cls_mlp(global_feat)
            smoothing_style_logits = self.smoothing_style_mlp(global_feat)

        pred_nocs = self.nocs_mlp(dense_feat)  # (B, N, 3)
        if self.training and gt_pc_nocs is not None:
            # use GT NOCS during training
            self.gt_nocs_ratio = max(self.min_gt_nocs_ratio,
                                     self.gt_nocs_ratio_decay_factor ** self.current_epoch)
            use_gt_nocs = torch.rand(1).item() < self.gt_nocs_ratio
        else:
            use_gt_nocs = False
        input_pc_nocs = gt_pc_nocs.transpose(1, 2) if use_gt_nocs else pred_nocs.detach().transpose(1, 2)  # (B, 3, N)
        dense_nocs_feat, _ = self.nocs_pointnet(input_pc_nocs)  # (B, C", N)
        num_pts = dense_feat_extra.shape[1]
        global_feat_expand = global_feat.unsqueeze(-1).expand(-1, -1, num_pts)  # (B, C, N)
        if self.use_nocs_for_dense_feat:
            dense_nocs_feat_cat = torch.cat([dense_nocs_feat, global_feat_expand], dim=1).transpose(1, 2)  # (B, N, C+C")
        else:
            dense_nocs_feat_cat = global_feat_expand.transpose(1, 2)
        dense_nocs_feat_cat = torch.cat([dense_nocs_feat_cat, dense_feat], dim=2)  # (B, N, C+C'+C")
        dense_nocs_feat_fuse = self.grasp_nocs_feat_mlp(dense_nocs_feat_cat)  # (B, N, C''')\
        
        keypoints = []
        for idx in range(self.num_keypoints):
            if self.detach_for_detector:
                offset = self.offset_mlp_list[idx](dense_feat.detach()) # stop gradient
                att = self.att_mlp_list[idx](dense_feat.detach()) # stop gradient
            else:
                offset = self.offset_mlp_list[idx](dense_feat)
                att = self.att_mlp_list[idx](dense_feat)
            att = torch.softmax(att, dim=1)
            keypoint = ((pc_xyz + offset) * att).sum(dim=1)  # (B, 3)
            keypoints.append(keypoint)
        keypoints = torch.stack(keypoints, dim=1)  # (B, num_keypoints, 3)

        return pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints

class RewardPredictionHead(pl.LightningModule):
    def __init__(self,
                 num_classes: int = 1,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 feature_dim: int = 240,
                 action_input_mlp_nn_channels: Tuple[int, int] = (120, 240),
                 action_output_mlp_nn_channels: Tuple[int, int] = (240, 120),
                 use_positional_encoding_in_attention_net: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.feature_dim = feature_dim

        self.model = AttentionNet(
            data_dim=feature_dim, # only for output dim
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_gripper_points=2,
            use_positional_encoding=use_positional_encoding_in_attention_net,
            action_input_mlp_nn_channels=action_input_mlp_nn_channels,
            action_output_mlp_nn_channels=action_output_mlp_nn_channels,
            enable_extra_outputting_dims=True
        )

        self.output_mlp = MLP_V2((feature_dim * 2, num_classes), transpose_input=True)

    def forward(self, action: torch.Tensor, context: torch.Tensor, context_pos: torch.Tensor) -> torch.tensor:
        B = action.size(0)
        # hack: fixed timestep
        timestep = torch.zeros(B, device=action.device)
        pred = self.model(action, timestep, context, context_pos)
        _, gp_dim, _ = pred.shape
        gp_dim //= 4
        pred1, pred2 = torch.chunk(pred, 2, dim=1)
        pred1 = pred1.reshape((-1, gp_dim, self.feature_dim * 2))
        pred2 = pred2.reshape((-1, gp_dim, self.feature_dim * 2))
        output1, output2 = self.output_mlp(pred1), self.output_mlp(pred2)
        return torch.cat([output1, output2], dim=2)

    def get_grasp_pair_scores(self, rankings: torch.Tensor):
        B, N = rankings.shape
        scores = torch.zeros(B, N, 2, device=rankings.device)
        weights = torch.zeros(B, N, 2, device=rankings.device)

        win_mask = (rankings == 0)

        scores[win_mask] = torch.tensor([1, 0], device=rankings.device, dtype=torch.float)
        weights[win_mask] = torch.tensor([1, 1], device=rankings.device, dtype=torch.float)

        lose_mask = (rankings == 1)

        scores[lose_mask] = torch.tensor([0, 1], device=rankings.device, dtype=torch.float)
        weights[lose_mask] = torch.tensor([1, 1], device=rankings.device, dtype=torch.float)

        return scores, weights

class PrimitiveDiffusion(pl.LightningModule):
    """
    Use Res-UNet3D as backbone, use Point cloud as input
    use Transformer to encode dense per-point feature
    use attention + offset to predict grasp points and release points
    predict K independent grasp-points in NOCS space  for fling action, use variety loss (nocs) for supervision
    factorized reward prediction
    """
    def __init__(self,
                 # sparse uned3d encoder params
                 sparse_unet3d_encoder_params,
                 # transformer params
                 transformer_params,
                 # action head params
                 state_head_params,
                 # diffusion head params
                 diffusion_head_params,
                 # reward head params
                 reward_prediction_head_params = None,
                 # hyper-params
                 rescale_nocs: bool = False,
                 use_multiple_poses: bool = False,
                 use_minsnr_reweight: bool = False,
                 use_matched_action_gt: bool = True,
                 # compatible with more tasks
                 valid_primitive_idx: int = ActionTypeDef.FLING.value,
                 valid_smoothed_values: tuple = (GeneralObjectState.to_int(GeneralObjectState.DISORDERED), GeneralObjectState.to_int(GeneralObjectState.ORGANIZED)),
                 valid_smoothing_style_values: tuple = (GarmentSmoothingStyle.to_int(GarmentSmoothingStyle.DOWN),
                                                        GarmentSmoothingStyle.to_int(GarmentSmoothingStyle.UP),
                                                        GarmentSmoothingStyle.to_int(GarmentSmoothingStyle.LEFT),
                                                        GarmentSmoothingStyle.to_int(GarmentSmoothingStyle.RIGHT)), # should be consecutive
                 gripper_points_idx: tuple = (0, 2), # (0, 1), (0, 1, 2, 3)
                 num_gripper_points: int = 2,
                 use_sym_loss: bool = True,
                 # loss weights
                 loss_cls_weight: float = 0.1,
                 loss_keypoint_weight: float = 1.0,
                 loss_nocs_weight: float = 100.0,
                 loss_diffusion_weight: float = 1.0,
                 loss_diffusion_finetune_weight: float = 1.0,
                 loss_diffusion_finetune_sft_equal_weight: float = 1.0,
                 loss_diffusion_finetune_sft_weight: float = 1.0,
                 loss_reward_prediction_weight: float = 1.0,
                 finetune_loss_type: str = 'dpo',
                 dpo_beta: float = 1000,
                 cpl_lambda: float = 1.0,
                 use_sft_for_equal_samples: bool = False,
                 use_sft_for_gt_data: bool = False,
                 # optimizer params
                 use_cos_lr: bool = False,
                 cos_t_max: int = 100,
                 init_lr: float = 1e-4,
                 # others
                 use_virtual_reward_for_inference: bool = True,
                 random_select_diffusion_action_pair_for_inference: bool = False,
                 manually_select_diffusion_action_pair_for_inference: bool = False,
                 use_dpo_reward_for_inference: bool = False,
                 use_reward_prediction_for_inference: bool = False,
                 dpo_reward_sample_num: int = 10,
                 reference_model_path: str = None,
                 enable_new_pipeline_finetune: bool = False,
                 enable_new_pipeline_supervised_classification_detection: bool = False,
                 enable_new_pipeline_reward_prediction: bool = False,
                 original_classification_model_path: str = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.rescale_nocs = rescale_nocs
        self.use_multiple_poses = use_multiple_poses
        self.use_minsnr_reweight = use_minsnr_reweight
        self.use_matched_action_gt = use_matched_action_gt
        # compatible with more tasks
        self.valid_primitive_idx = valid_primitive_idx
        self.valid_smoothed_values = valid_smoothed_values
        self.valid_smoothing_style_values = valid_smoothing_style_values
        self.gripper_points_idx = gripper_points_idx
        self.num_gripper_points = num_gripper_points
        self.use_sym_loss = use_sym_loss
        # loss weights
        self.loss_cls_weight = loss_cls_weight
        self.loss_keypoint_weight = loss_keypoint_weight
        self.loss_nocs_weight = loss_nocs_weight
        self.loss_diffusion_weight = loss_diffusion_weight
        self.loss_reward_prediction_weight = loss_reward_prediction_weight
        self.loss_diffusion_finetune_weight = loss_diffusion_finetune_weight
        self.loss_diffusion_finetune_sft_euqal_weight = loss_diffusion_finetune_sft_equal_weight
        self.loss_diffusion_finetune_sft_weight = loss_diffusion_finetune_sft_weight
        if finetune_loss_type is not None:
            assert finetune_loss_type in ['dpo', 'cpl'], "Only support DPO and CPL loss"
        else:
            assert use_sft_for_gt_data or enable_new_pipeline_reward_prediction, \
                "Finetune loss type should be provided when not using SFT for GT data and not enable bew pipeline reward prediction"
        self.finetune_loss_type = finetune_loss_type
        self.dpo_beta = dpo_beta
        self.cpl_lambda = cpl_lambda
        if use_sft_for_equal_samples:
            assert finetune_loss_type in ['dpo', 'cpl'], "Only support DPO and CPL loss when using SFT for equal samples"
        self.use_sft_for_equal_samples = use_sft_for_equal_samples
        self.use_sft_for_gt_data = use_sft_for_gt_data
        # optimizer params
        self.use_cos_lr = use_cos_lr
        self.cos_t_max = cos_t_max
        self.init_lr = init_lr
        # others
        self.use_virtual_reward_for_inference = use_virtual_reward_for_inference
        if use_dpo_reward_for_inference:
            assert not use_virtual_reward_for_inference, "DPO reward inference is not compatible with virtual reward inference"
            assert not use_reward_prediction_for_inference, "DPO reward inference is not compatible with reward prediction inference"
            assert not random_select_diffusion_action_pair_for_inference, "DPO reward inference is not compatible with randomly select action pair"
        self.use_dpo_reward_for_inference = use_dpo_reward_for_inference
        self.dpo_reward_sample_num = dpo_reward_sample_num
        if use_reward_prediction_for_inference:
            assert not use_virtual_reward_for_inference, "reward prediction inference is not compatible with virtual reward inference"
            assert not use_dpo_reward_for_inference, "reward prediction inference is not compatible with DPO reward inference"
            assert not random_select_diffusion_action_pair_for_inference, "reward prediction inference is not compatible with randomly select action pair"
        self.use_reward_prediction_for_inference = use_reward_prediction_for_inference
        if random_select_diffusion_action_pair_for_inference:
            assert not use_virtual_reward_for_inference, "randomly select action pair is not compatible with virtual reward inference"
            assert not use_dpo_reward_for_inference, "randomly select action pair is not compatible with DPO reward inference"
            assert not use_reward_prediction_for_inference, "randomly select action pair is not compatible with reward prediction inference"
        self.random_select_diffusion_action_pair_for_inference = random_select_diffusion_action_pair_for_inference
        if manually_select_diffusion_action_pair_for_inference:
            if not random_select_diffusion_action_pair_for_inference:
                logger.warning("manually select action pair is not enabled when randomly select action pair is disabled")
            manually_select_diffusion_action_pair_for_inference = False
        self.manually_select_diffusion_action_pair_for_inference = manually_select_diffusion_action_pair_for_inference

        self.backbone = SparseResUNet(**sparse_unet3d_encoder_params)
        self.transformer = Transformer(**transformer_params)
        self.state_head = StateHead(**state_head_params)
        self.diffusion_head = DiffusionHead(**diffusion_head_params)

        self.sigmoid = nn.Sigmoid()
        
        if enable_new_pipeline_finetune:
            assert reference_model_path is not None, "reference model path should be provided for new pipeline finetuning"
        self.enable_new_pipeline_finetune = enable_new_pipeline_finetune

        if enable_new_pipeline_supervised_classification_detection:
            assert not enable_new_pipeline_finetune, "new pipeline supervised smoothing style is not compatible with new pipeline finetuning"
            if original_classification_model_path is not None and osp.exists(original_classification_model_path):
                checkpoint_dir = osp.join(original_classification_model_path, 'checkpoints')
                checkpoint_path = osp.join(checkpoint_dir, 'last.ckpt')
                classification_original_model = PrimitiveDiffusion.load_from_checkpoint(checkpoint_path, strict=False)
                self.load_state_dict(classification_original_model.state_dict(), strict=False)
            else:
                logger.warning("original classification model path is not provided or not exists, train from scratch for new pipeline supervised classification detection")
        self.enable_new_pipeline_supervised_classification_detection = enable_new_pipeline_supervised_classification_detection

        if enable_new_pipeline_reward_prediction:
            assert enable_new_pipeline_finetune, "new pipeline reward prediction is only compatible with new pipeline finetuning"
            self.reward_head = RewardPredictionHead(**reward_prediction_head_params)
        self.enable_new_pipeline_reward_prediction = enable_new_pipeline_reward_prediction

        # reference model for new pipeline finetuning
        if reference_model_path is not None:
            checkpoint_dir = osp.join(reference_model_path, 'checkpoints')
            checkpoint_path = osp.join(checkpoint_dir, 'last.ckpt')
            reference_model = PrimitiveDiffusion.load_from_checkpoint(checkpoint_path, strict=True)
            reference_model.eval()
            reference_model.requires_grad_(False)
            
            self.load_state_dict(reference_model.state_dict(), strict=True)
            self.reference_model = reference_model
            
            self.sync_reference_model_settings()
        else:
            self.reference_model = None    
        
        if use_dpo_reward_for_inference or use_reward_prediction_for_inference:
            assert self.reference_model is not None, "reference model should be provided for DPO reward inference"

    def sync_reference_model_settings(self):
        self.reference_model.use_virtual_reward_for_inference = self.use_virtual_reward_for_inference
        self.reference_model.random_select_diffusion_action_pair_for_inference = self.random_select_diffusion_action_pair_for_inference
        self.reference_model.manually_select_diffusion_action_pair_for_inference = self.manually_select_diffusion_action_pair_for_inference

        self.reference_model.use_dpo_reward_for_inference = self.use_dpo_reward_for_inference
        self.reference_model.use_reward_prediction_for_inference = self.use_reward_prediction_for_inference
        self.reference_model.dpo_reward_sample_num = self.dpo_reward_sample_num

        self.reference_model.diffusion_head.num_of_grasp_points = self.diffusion_head.num_of_grasp_points
        self.reference_model.state_head.num_pred_fling_candidates = self.state_head.num_pred_fling_candidates

        self.reference_model.diffusion_head.scheduler_type = self.diffusion_head.scheduler_type
        self.reference_model.diffusion_head.num_inference_steps = self.diffusion_head.num_inference_steps
        self.reference_model.diffusion_head.ddim_eta = self.diffusion_head.ddim_eta

    def configure_optimizers(self):
        if self.enable_new_pipeline_supervised_classification_detection:
            freeze_net = [self.diffusion_head]
            for net in freeze_net:
                for param in net.parameters():
                    param.requires_grad = False
            all_parameters = [{"params": self.state_head.parameters()},
                              {"params": self.backbone.parameters()},
                              {"params": self.transformer.parameters()}]
        else:
            optim_groups = self.diffusion_head.model.get_optim_groups(weight_decay=self.diffusion_head.weight_decay)
            all_parameters = optim_groups + [{"params": self.state_head.parameters()},
                                            {"params": self.backbone.parameters()},
                                            {"params": self.transformer.parameters()}]
            if self.enable_new_pipeline_reward_prediction:
                all_parameters += [{"params": self.reward_head.parameters()}]
        optimizer = torch.optim.AdamW(all_parameters, lr=self.init_lr, betas=[0.9, 0.95])
        if self.use_cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cos_t_max)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return [optimizer], [scheduler]

    def forward(self, coords: torch.Tensor, feat: torch.Tensor, pc_xyz: torch.Tensor, gt_pc_nocs: torch.Tensor = None):
        input = ME.SparseTensor(feat, coordinates=coords)
        dense_feat = self.backbone(input)  # (B*N, C)
        dense_feat_att = self.transformer(dense_feat, pc_xyz.view(-1, 3))  # (B, N, C)
        # TODO: add rotation angle prediction
        pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints = \
            self.state_head(pc_xyz, dense_feat_att, gt_pc_nocs=gt_pc_nocs)
        return dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints

    @staticmethod
    def bce_loss(prediction, target, weights=None):
        if weights is None:
            weights = 1.0
        valid_count = max(weights[:, 0].sum().item(), 1.0)
        return (weights * nn.BCEWithLogitsLoss(reduction='none')(prediction, target.float())).mean(dim=1).sum() / valid_count


    @staticmethod
    def sym_grasp_mse_variety_cls_err(pred_grasp_nocs, pred_pc_nocs,
                                         pc_xyz, xyz_target, nocs_target, weights=None,
                                         use_xyz_variety_loss=False, alpha=30.0):
        """

        :param pred_grasp_nocs:  (B, K, 3)
        :param pred_grasp_score: (B, K)
        :param pred_pc_nocs: (B, N, 3)
        :param pc_xyz: (B, N, 3)
        :param xyz_target: (B, 2, 3)
        :param nocs_target: (B, 2, 3)
        :param weights:
        :param use_xyz_variety_loss: bool, whether to calculate xyz variety loss
        :param alpha: float, the weight for exponential function in nocs-distance weight calculation
        :return:
        """
        B = pred_grasp_nocs.shape[0]
        K = pred_grasp_nocs.shape[1]
        N = pc_xyz.shape[1]
        device = pred_grasp_nocs.device
        if weights is None:
            weights = 1.0
        valid_count = max(weights.sum().item(), 1.0)

        # nocs variety loss
        nocs_metric = torch.nn.MSELoss(reduction='none')
        left_target_nocs = nocs_target[:, 0, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)
        right_target_nocs = nocs_target[:, 1, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)

        left_grasp_loss_nocs = nocs_metric(pred_grasp_nocs, left_target_nocs).mean(dim=-1)  # (B, K)
        right_grasp_loss_nocs = nocs_metric(pred_grasp_nocs, right_target_nocs).mean(dim=-1)  # (B, K)

        left_variety_loss_nocs, left_target_idxs_nocs = torch.min(left_grasp_loss_nocs, dim=1)  # (B, )
        right_variety_loss_nocs, right_target_idxs_nocs = torch.min(right_grasp_loss_nocs, dim=1)  # (B, )
        variety_loss_nocs = (left_variety_loss_nocs + right_variety_loss_nocs) / 2.0  # (B, )
        loss_grasp_variety_nocs = (variety_loss_nocs * weights).sum() / valid_count

        if use_xyz_variety_loss:
            # xyz variety loss with nocs-distance as weights
            xyz_metric = torch.nn.MSELoss(reduction='none')
            left_target_xyz = xyz_target[:, 0, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)
            right_target_xyz = xyz_target[:, 1, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)

            pred_grasp_nocs_expand = pred_grasp_nocs.unsqueeze(2).expand(-1, -1, N, -1)  # (B, K, N, 3)
            # detach pred_grasp_nocs to avoid gradient flow back to pred_grasp_nocs
            pred_pc_nocs_expand = pred_pc_nocs.detach().unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, 3)
            nocs_distance = torch.norm(pred_grasp_nocs_expand - pred_pc_nocs_expand, dim=-1)  # (B, K, N)
            nocs_distance_weight = torch.exp(-alpha * nocs_distance)  # (B, K, N)
            normalized_nocs_distance_weight = nocs_distance_weight / nocs_distance_weight.sum(dim=-1, keepdim=True) + 1e-6 # (B, K, N)
            
            pc_xyz_expand = pc_xyz.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, 3)
            pred_grasp_xyz = torch.sum(pc_xyz_expand * normalized_nocs_distance_weight.unsqueeze(-1), dim=2)  # (B, K, 3)

            left_grasp_loss_xyz = xyz_metric(pred_grasp_xyz, left_target_xyz).mean(dim=-1)  # (B, K)
            right_grasp_loss_xyz = xyz_metric(pred_grasp_xyz, right_target_xyz).mean(dim=-1)  # (B, K)
            batch_range = torch.arange(B, device=device)  # (B, )
            left_variety_loss_xyz = left_grasp_loss_xyz[batch_range, left_target_idxs_nocs]  # (B, )
            right_variety_loss_xyz = right_grasp_loss_xyz[batch_range, right_target_idxs_nocs]  # (B, )
            variety_loss_xyz = (left_variety_loss_xyz + right_variety_loss_xyz) / 2.0  # (B, )
            loss_grasp_variety_xyz = (variety_loss_xyz * weights).sum() / valid_count

            return loss_grasp_variety_nocs.detach(), loss_grasp_variety_xyz.detach()
        else:
            return loss_grasp_variety_nocs.detach(), torch.zeros_like(loss_grasp_variety_nocs)

    @staticmethod
    def sym_nocs_huber_loss(prediction: torch.Tensor, target: torch.Tensor, rescale_nocs: bool = False) -> torch.Tensor:
        metric = torch.nn.HuberLoss(delta=0.1, reduction='none')
        sym_target = target.clone()
        # symmetric target: 180 deg rotation around z-axis in NOCS space
        if rescale_nocs:
            sym_target[:, :, 0] = - sym_target[:, :, 0]
            sym_target[:, :, 1] = - sym_target[:, :, 1]
        else:
            sym_target[:, :, 0] = 1.0 - sym_target[:, :, 0]
            sym_target[:, :, 1] = 1.0 - sym_target[:, :, 1]
        loss = torch.minimum(metric(prediction, target).mean((1, 2)),
                             metric(prediction, sym_target).mean((1, 2))
                             ).mean()
        return loss

    @staticmethod
    def sym_diffusion_mse_loss(action_gt: torch.Tensor,
                               action_pred: torch.Tensor,
                               weights: torch.Tensor = None,
                               minsnr_weights: torch.Tensor = None,
                               use_sym_loss: bool = True) -> torch.Tensor:
        metric = torch.nn.MSELoss(reduction='none')
        
        B, K, D = action_gt.shape
        action_gt = action_gt.reshape(B, K//2, 2, D)
        action_pred = action_pred.reshape(B, K//2, 2, D)

        if use_sym_loss:
            sym_action_gt = action_gt.clone()
            sym_action_gt[:, :, 0, :] = action_gt[:, :, 1, :]
            sym_action_gt[:, :, 1, :] = action_gt[:, :, 0, :]

            loss = torch.minimum(metric(action_pred, action_gt).mean((2, 3)),
                                metric(action_pred, sym_action_gt).mean((2, 3))
                                ).mean(1)
        else:
            loss = metric(action_pred, action_gt).mean((1, 2, 3))
        
        if weights is None:
            weights = torch.tensor(1.0)
        valid_count = max(weights.sum().item(), 1.0)
        
        if minsnr_weights is None:
            minsnr_weights = torch.tensor(1.0)

        return (loss * weights).sum() / valid_count, (loss * weights * minsnr_weights).sum() / valid_count
    
    @staticmethod
    def epsilon_diffusion_mse_loss(noise_gt: torch.Tensor,
                               action_pred: torch.Tensor,
                               weights: torch.Tensor = None,
                               minsnr_weights: torch.Tensor = None) -> torch.Tensor:
        metric = torch.nn.MSELoss(reduction='none')
        
        B, K, D = noise_gt.shape
        noise_gt = noise_gt.reshape(B, K//2, 2, D)
        action_pred = action_pred.reshape(B, K//2, 2, D)

        loss = metric(action_pred, noise_gt).mean((1, 2, 3))
        
        if weights is None:
            weights = torch.tensor(1.0)
        valid_count = max(weights.sum().item(), 1.0)
        
        if minsnr_weights is None:
            minsnr_weights = torch.tensor(1.0)

        return (loss * weights).sum() / valid_count, (loss * weights * minsnr_weights).sum() / valid_count
    
    @staticmethod
    def diffusion_dpo_loss(action_gt: torch.Tensor,
                            action_pred: torch.Tensor,
                            action_pred_reference: torch.Tensor,
                            rankings: torch.Tensor,
                            weights: torch.Tensor = None,
                            minsnr_weights: torch.Tensor = None,
                            beta: float = 1000) -> torch.Tensor:
        
        metric = torch.nn.MSELoss(reduction='none')
        sigmoid = torch.nn.Sigmoid()
        
        win_mask = (rankings == 0) # win
        win_weights = win_mask.float()
        
        lose_mask = (rankings == 1) # lose
        lose_weights = lose_mask.float() * -1
        
        equal_mask = (rankings == 2) # equal
        equal_weights = equal_mask.float()
        
        action_gt_1, action_gt_2 = action_gt.chunk(2, dim=1)
        action_pred_1, action_pred_2 = action_pred.chunk(2, dim=1)
        action_pred_reference_1, action_pred_reference_2 = action_pred_reference.chunk(2, dim=1)
        
        elbo1 = metric(action_pred_1, action_gt_1).mean((2, 3))
        elbo2 = metric(action_pred_2, action_gt_2).mean((2, 3))
        elbo1_reference = metric(action_pred_reference_1, action_gt_1).mean((2, 3))
        elbo2_reference = metric(action_pred_reference_2, action_gt_2).mean((2, 3))

        loss = (elbo1 - elbo1_reference) - (elbo2 - elbo2_reference)
        loss = loss * win_weights + loss * lose_weights
        
        if minsnr_weights is None:
            minsnr_weights = torch.tensor(1.0)
        else:
            minsnr_weights = minsnr_weights.unsqueeze(1)
            
        loss_constant_weighted = - sigmoid( - beta * loss)
        loss_snr_weighted = - sigmoid( - beta * minsnr_weights * loss)
        
        loss_sft_constant_weighted = equal_weights * (elbo1 + elbo2) / 2
        loss_sft_snr_weighted = equal_weights * (elbo1 + elbo2) / 2
        
        valid_mask = win_mask | lose_mask
        valid_weight = valid_mask.float()
        if weights is None:
            valid_weight = valid_weight * weights.unsqueeze(1)
        valid_count = max(valid_weight.sum().item(), 1.0)
        
        valid_weight_equal = equal_weights.clone()
        if weights is None:
            valid_weight_equal = valid_weight_equal * weights.unsqueeze(1)
        valid_count_equal = max(valid_weight_equal.sum().item(), 1.0)
        
        return (loss_constant_weighted * valid_weight).sum() / valid_count, (loss_snr_weighted * valid_weight).sum() / valid_count, \
            (loss_sft_constant_weighted * valid_weight_equal).sum() / valid_count_equal, (loss_sft_snr_weighted * valid_weight_equal).sum() / valid_count_equal
    
    @staticmethod
    def diffusion_cpl_loss(action_gt: torch.Tensor,
                            action_pred: torch.Tensor,
                            rankings: torch.Tensor,
                            weights: torch.Tensor = None,
                            minsnr_weights: torch.Tensor = None,
                            cpl_lambda: float = 1.0) -> torch.Tensor:
        
        metric = torch.nn.MSELoss(reduction='none')
        
        win_mask = (rankings == 0) # win
        win_weights = win_mask.float()
        
        lose_mask = (rankings == 1) # lose
        lose_weights = lose_mask.float() * -1
        
        equal_mask = (rankings == 2) # equal
        equal_weights = equal_mask.float()
        
        action_gt_1, action_gt_2 = action_gt.chunk(2, dim=1)
        action_pred_1, action_pred_2 = action_pred.chunk(2, dim=1)
        
        elbo1 = metric(action_pred_1, action_gt_1).mean((2, 3))
        elbo2 = metric(action_pred_2, action_gt_2).mean((2, 3))
        
        if minsnr_weights is None:
            minsnr_weights = torch.tensor(1.0)
        else:
            minsnr_weights = minsnr_weights.unsqueeze(1)
        
        elbo_perfer = elbo1 * win_weights - elbo2 * lose_weights
        elbo_not_perfer = - elbo1 * lose_weights + elbo2 * win_weights
        #TODO: support using cpl lambda for regularization
        loss_constant_weighted = - torch.log(torch.exp(-elbo_perfer) / (torch.exp(-elbo_perfer) + torch.exp(- elbo_not_perfer)))
        loss_snr_weighted = - torch.log(torch.exp(-minsnr_weights * elbo_perfer) / (torch.exp(-minsnr_weights * elbo_perfer) + torch.exp(-minsnr_weights * elbo_not_perfer)))
        
        loss_sft_constant_weighted = equal_weights * (elbo1 + elbo2) / 2
        loss_sft_snr_weighted = equal_weights * (elbo1 + elbo2) / 2
        
        valid_mask = win_mask | lose_mask
        valid_weight = valid_mask.float()
        if weights is None:
            valid_weight = valid_weight * weights.unsqueeze(1)
        valid_count = max(valid_weight.sum().item(), 1.0)
        
        valid_weight_equal = equal_weights.clone()
        if weights is None:
            valid_weight_equal = valid_weight_equal * weights.unsqueeze(1)
        valid_count_equal = max(valid_weight_equal.sum().item(), 1.0)
        
        return (loss_constant_weighted * valid_weight).sum() / valid_count, (loss_snr_weighted * valid_weight).sum() / valid_count, \
            (loss_sft_constant_weighted * valid_weight_equal).sum() / valid_count_equal, (loss_sft_snr_weighted * valid_weight_equal).sum() / valid_count_equal
            
    @staticmethod
    def ranking_loss(pred: torch.Tensor, gt_score: torch.Tensor, weights = None) -> torch.Tensor:
        """
        Input:
            pred: (B, K, 2) Tensor
            gt_score: (B, K, 2) Tensor
        """
        # pred_all = torch.stack([pred1, pred2], dim=-1)  # (B, K, 2)
        if weights is None:
            weights = torch.ones_like(gt_score)
        pred = pred * weights
        pred_all = pred
        gt_score = gt_score * weights
        # TODO: change to BCEWithLogitsLoss
        pred_score_all = F.softmax(pred_all, dim=-1)  # (B, K, 2)
        loss = - (gt_score[:, :, 0] * torch.log(pred_score_all[:, :, 0]) +
                  gt_score[:, :, 1] * torch.log(pred_score_all[:, :, 1]))
        loss = loss.mean()
        return loss

    def expand_to(self, x):
        return x.view(-1, 1)

    def get_diffusion_prediction_from_model(self, model, coords, feat, pc_xyz, pc_nocs, action_xyz_gt, action_nocs_gt, given_timesteps=None, given_noise=None):
        dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints = \
            model.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)
    
        fling_action_gt, fling_action_pred, fling_action_xyz_pred, fling_action_nocs_pred, timesteps, noise = self.diffusion_head.diffuse_denoise(
            action_xyz_gt=action_xyz_gt,
            action_nocs_gt=action_nocs_gt,
            context=dense_nocs_feat_fuse,
            pc_xyz=pc_xyz,
            pc_nocs=pc_nocs if use_gt_nocs else pred_nocs,
            use_matched_action_gt=self.use_matched_action_gt,
            timesteps=given_timesteps,
            noise=given_noise,
            replicate_action=False
        )
        
        return fling_action_gt, fling_action_pred, timesteps, noise
        
    def forward_finetune_loss(self, input_batch: tuple):
        coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, rotation_cls, \
            reward, primitive_index_batch, smoothed_label, smoothing_style_label, folding_score, folding_step, \
            grasp_point_pair1, grasp_point_pair2, grasp_pair_scores, grasp_point_nocs_pair1, grasp_point_nocs_pair2, rankings = \
            tuple(x.to(self.device) for x in input_batch)
            
        batch_size = pc_xyz.shape[0]
        weights = torch.zeros(batch_size, device=self.device)  # (B, )
        weights[primitive_index_batch == self.valid_primitive_idx] = 1.0
        
        if self.finetune_loss_type == 'dpo' or self.finetune_loss_type == 'cpl':
            action_xyz_gt = torch.cat([grasp_point_pair1.permute(0, 2, 1, 3), grasp_point_pair2.permute(0, 2, 1, 3)], dim=1)
            action_nocs_gt = torch.cat([grasp_point_nocs_pair1.permute(0, 2, 1, 3), grasp_point_nocs_pair2.permute(0, 2, 1, 3)], dim=1)
            B, _, _, D = action_xyz_gt.shape
            action_xyz_gt = action_xyz_gt.reshape(B, -1, D)
            B, _, _, D = action_nocs_gt.shape
            action_nocs_gt = action_nocs_gt.reshape(B, -1, D)
            
            fling_action_gt, fling_action_pred, timesteps, noise = self.get_diffusion_prediction_from_model(
                self, coords, feat, pc_xyz, pc_nocs, action_xyz_gt, action_nocs_gt
            )
            B, K, D = fling_action_gt.shape
            fling_action_gt = fling_action_gt.reshape(B, K//self.num_gripper_points, self.num_gripper_points, D)
            fling_action_pred = fling_action_pred.reshape(B, K//self.num_gripper_points, self.num_gripper_points, D)
            
            if self.use_minsnr_reweight:
                minsnr_weights = self.diffusion_head.get_minsnr_weights(timesteps)
            else:
                minsnr_weights = None
            
            if self.finetune_loss_type == 'dpo':
                _, fling_action_pred_reference, timesteps, noise = self.get_diffusion_prediction_from_model(
                    self.reference_model, coords, feat, pc_xyz, pc_nocs, action_xyz_gt, action_nocs_gt, given_timesteps=timesteps, given_noise=noise
                )
                fling_action_pred_reference = fling_action_pred_reference.reshape(B, K//self.num_gripper_points, self.num_gripper_points, D)
                
                loss_diffusion_finetune, loss_diffusion_finetune_weighted, loss_sft_equal, loss_sft_equal_weighted = self.diffusion_dpo_loss(fling_action_gt,
                                                                                                                            fling_action_pred,
                                                                                                                            fling_action_pred_reference,
                                                                                                                            rankings,
                                                                                                                            weights=weights,
                                                                                                                            minsnr_weights=minsnr_weights,
                                                                                                                            beta=self.dpo_beta)        

            elif self.finetune_loss_type == 'cpl':
                loss_diffusion_finetune, loss_diffusion_finetune_weighted, loss_sft_equal, loss_sft_equal_weighted = self.diffusion_cpl_loss(fling_action_gt,
                                                                                                                                fling_action_pred,
                                                                                                                                rankings,
                                                                                                                                weights=weights,
                                                                                                                                minsnr_weights=minsnr_weights,
                                                                                                                                cpl_lambda=self.cpl_lambda)
                    
            # ------------------------------------------
            # import open3d as o3d
            # geometries = []
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc_xyz[0].detach().cpu().numpy())
            # geometries.append(pcd)
            # grasp_points = fling_action_gt[0].detach().cpu().numpy()
            # grasp_points = grasp_points.reshape(-1, 3)
            # for i in range(2):
            #     pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            #     pcd.translate(grasp_points[i])
            #     geometries.append(pcd)
            # o3d.visualization.draw_geometries(geometries)
            # ------------------------------------------
        else:
            loss_diffusion_finetune = None
            loss_diffusion_finetune_weighted = None

        if self.enable_new_pipeline_reward_prediction:
            action_xyz_gt = torch.cat([grasp_point_pair1.permute(0, 2, 1, 3), grasp_point_pair2.permute(0, 2, 1, 3)], dim=1)
            B, _, _, D = action_xyz_gt.shape
            action_xyz_gt = action_xyz_gt.reshape(B, -1, D)

            _, _, _, dense_nocs_feat_fuse, _, _, _ = \
                self.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)

            reward_pred = self.reward_head(action_xyz_gt, dense_nocs_feat_fuse, pc_xyz)
            grasp_pair_scores_unsoften, grasp_pair_weights = self.reward_head.get_grasp_pair_scores(rankings)
            loss_reward_prediction = self.ranking_loss(reward_pred, grasp_pair_scores_unsoften, grasp_pair_weights)
        else:
            loss_reward_prediction = None

        if self.use_sft_for_gt_data:  
            dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints = \
            self.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)
            
            fling_action_gt, fling_action_pred, fling_action_xyz_pred, fling_action_nocs_pred, timesteps, noise = self.diffusion_head.diffuse_denoise(
                action_xyz_gt=gripper_points[:, self.gripper_points_idx, :],
                action_nocs_gt=grasp_points_nocs,
                context=dense_nocs_feat_fuse,
                pc_xyz=pc_xyz,
                pc_nocs=pc_nocs if use_gt_nocs else pred_nocs,
                use_matched_action_gt=self.use_matched_action_gt,
            )
                     
            if self.use_minsnr_reweight:
                minsnr_weights = self.diffusion_head.get_minsnr_weights(timesteps)
            else:
                minsnr_weights = None
            if self.diffusion_head.noise_scheduler.config.prediction_type == 'sample':
                loss_sft, loss_sft_weighted = self.sym_diffusion_mse_loss(fling_action_gt, fling_action_pred, weights=weights, minsnr_weights=minsnr_weights, use_sym_loss=self.use_sym_loss)
            else:
                loss_sft, loss_sft_weighted = self.epsilon_diffusion_mse_loss(fling_action_gt, fling_action_pred, weights=weights, minsnr_weights=minsnr_weights)
        else:
            loss_sft = None
            loss_sft_weighted = None
            
        loss_dict = dict()
        loss_dict.update({'lr': self.optimizers().optimizer.param_groups[0]['lr']})
        
        if loss_diffusion_finetune is not None and loss_diffusion_finetune_weighted is not None:
            loss_diffusion_finetune *= self.loss_diffusion_finetune_weight
            loss_diffusion_finetune_weighted *= self.loss_diffusion_finetune_weight
            if self.use_minsnr_reweight:
                loss_diffusion_finetune = loss_diffusion_finetune.detach()
            else:
                loss_diffusion_finetune_weighted = loss_diffusion_finetune_weighted.detach()
            
            loss_dict.update({'loss_diffusion_finetune': loss_diffusion_finetune,
                                'loss_diffusion_finetune_weighted': loss_diffusion_finetune_weighted})

        if loss_reward_prediction is not None:
            loss_reward_prediction *= self.loss_reward_prediction_weight
            loss_dict.update({'loss_reward_prediction': loss_reward_prediction})

        if self.use_sft_for_equal_samples:
            loss_sft_equal *= self.loss_diffusion_finetune_sft_euqal_weight
            loss_sft_equal_weighted *= self.loss_diffusion_finetune_sft_euqal_weight
            if self.use_minsnr_reweight:
                loss_sft_equal = loss_sft_equal.detach()
            else:
                loss_sft_equal_weighted = loss_sft_equal_weighted.detach()
            loss_dict.update({'loss_sft_equal': loss_sft_equal,
                                'loss_sft_equal_weighted': loss_sft_equal_weighted})
        
        if self.use_sft_for_gt_data:
            loss_sft *= self.loss_diffusion_finetune_sft_weight
            loss_sft_weighted *= self.loss_diffusion_finetune_sft_weight
            if self.use_minsnr_reweight:
                loss_sft = loss_sft.detach()
            else:
                loss_sft_weighted = loss_sft_weighted.detach()
            loss_dict.update({'loss_sft': loss_sft,
                                'loss_sft_weighted': loss_sft_weighted})
        
        loss_all = torch.zeros_like(loss_dict['loss_diffusion_finetune'] if 'loss_diffusion_finetune' in loss_dict else loss_dict['loss_sft'] if 'loss_sft' in loss_dict else loss_reward_prediction)
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item
        loss_dict['finetune_loss'] = loss_all
        return loss_dict
    
    def _calculate_smoothed_loss_and_metrics(self, pred_smoothed_logits, smoothed_label):
        # classification loss: is_smoothed
        smoothed_valid_mask = torch.zeros_like(smoothed_label, dtype=torch.bool)
        for v in self.valid_smoothed_values: smoothed_valid_mask = smoothed_valid_mask | (smoothed_label == v)
        smoothed_valid_mask = smoothed_valid_mask.float()
        target_smoothed = (smoothed_label == GeneralObjectState.to_int(GeneralObjectState.ORGANIZED)).float()
        loss_smoothed = self.loss_cls_weight * (nn.BCEWithLogitsLoss(
            reduction='none')(pred_smoothed_logits, target_smoothed.unsqueeze(1)) * smoothed_valid_mask.unsqueeze(1))
        loss_smoothed = loss_smoothed.sum() / max(smoothed_valid_mask.sum().item(), 1.0)
        
        # calculate metrics
        pred_smoothed = (torch.sigmoid(pred_smoothed_logits) > 0.5)[:, 0]
        smoothed_accuracy = (pred_smoothed == target_smoothed).float() * smoothed_valid_mask
        smoothed_accuracy = smoothed_accuracy.sum() / max(smoothed_valid_mask.sum().item(), 1.0)
        
        smoothed_false_positive = ((pred_smoothed == 1) & (target_smoothed == 0)).float() * smoothed_valid_mask
        smoothed_false_positive = smoothed_false_positive.sum() / max(smoothed_valid_mask.sum().item(), 1.0)
        
        smoothed_false_negative = ((pred_smoothed == 0) & (target_smoothed == 1)).float() * smoothed_valid_mask
        smoothed_false_negative = smoothed_false_negative.sum() / max(smoothed_valid_mask.sum().item(), 1.0)
        
        if target_smoothed.sum().item() == 0 or target_smoothed.sum().item() == target_smoothed.shape[0]:
            smoothed_auc = 0.5
        else:
            smoothed_auc = roc_auc_score(target_smoothed.detach().cpu().numpy(), torch.sigmoid(pred_smoothed_logits).detach().cpu().numpy())
        
        return loss_smoothed, smoothed_accuracy, smoothed_false_positive, smoothed_false_negative, smoothed_auc

    def _calculate_smoothing_style_loss_and_metrics(self, pred_smoothing_style_logits, smoothing_style_label):
        # classification loss: smoothing style
        smoothing_style_valid_mask = torch.zeros_like(smoothing_style_label, dtype=torch.bool)
        for v in self.valid_smoothing_style_values: smoothing_style_valid_mask = smoothing_style_valid_mask | (smoothing_style_label == v)
        smoothing_style_label[~smoothing_style_valid_mask] = 0 # to avoid error
        smoothing_style_valid_mask = smoothing_style_valid_mask.float()
        loss_smoothing_style = self.loss_cls_weight * (nn.CrossEntropyLoss(
            reduction='none')(pred_smoothing_style_logits, smoothing_style_label) * smoothing_style_valid_mask)
        loss_smoothing_style = loss_smoothing_style.sum() / max(smoothing_style_valid_mask.sum().item(), 1.0)
        
        # calculate metrics
        pred_smoothing_style = torch.argmax(pred_smoothing_style_logits, dim=1)
        smoothing_style_accuracy = (pred_smoothing_style == smoothing_style_label).float() * smoothing_style_valid_mask
        smoothing_style_accuracy = smoothing_style_accuracy.sum() / max(smoothing_style_valid_mask.sum().item(), 1.0)

        return loss_smoothing_style, smoothing_style_accuracy
    
    def _calculate_keypoint_detection_loss_and_metrics(self, pred_keypoints, keypoints, smoothed_label):
        smoothed_mask = (smoothed_label == GeneralObjectState.to_int(GeneralObjectState.ORGANIZED)).float()
        
        # classification loss: keypoint detection
        loss_keypoint_mse = self.loss_keypoint_weight * ((nn.MSELoss(
            reduction='none')(pred_keypoints, keypoints)).mean((1, 2)) * smoothed_mask)
        loss_keypoint_mse = loss_keypoint_mse.sum() / max(smoothed_mask.sum().item(), 1.0)
        
        # calculate metrics
        keypoint_err = torch.norm(pred_keypoints - keypoints, dim=-1)
        keypoint_err = keypoint_err.mean(1) * smoothed_mask
        keypoint_err = keypoint_err.sum() / max(smoothed_mask.sum().item(), 1.0)
        
        return loss_keypoint_mse, keypoint_err
        
    def forward_supervised_loss(self, input_batch: tuple):
        coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, keypoints, rotation_cls, \
            reward, primitive_index_batch, smoothed_label, smoothing_style_label, folding_score, folding_step = \
            tuple(x.to(self.device) for x in input_batch)
        if self.use_multiple_poses:
            multiple_gripper_points = gripper_points.clone()
            multiple_grasp_points_nocs = grasp_points_nocs.clone()
            chosen_pose_idx = np.random.randint(0, gripper_points.shape[1])
            gripper_points = multiple_gripper_points[:, chosen_pose_idx]
            grasp_points_nocs = multiple_grasp_points_nocs[:, chosen_pose_idx]
            rotation_cls = rotation_cls[:, chosen_pose_idx]
        # network forward
        dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, pred_smoothed_logits, pred_smoothing_style_logits, pred_keypoints = \
            self.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)

        # grasp loss for NOCS coordintes of grasp points for fling action
        batch_size = pc_xyz.shape[0]
        weights = torch.zeros(batch_size, device=self.device)  # (B, )
        weights[primitive_index_batch == self.valid_primitive_idx] = 1.0
        
        fling_action_gt, fling_action_pred, fling_action_xyz_pred, fling_action_nocs_pred, timesteps, noise = self.diffusion_head.diffuse_denoise(
            action_xyz_gt=gripper_points[:, self.gripper_points_idx, :],
            action_nocs_gt=grasp_points_nocs,
            context=dense_nocs_feat_fuse,
            pc_xyz=pc_xyz,
            pc_nocs=pc_nocs if use_gt_nocs else pred_nocs,
            use_matched_action_gt=self.use_matched_action_gt,
            multiple_action_xyz_gt=multiple_gripper_points[:, :, self.gripper_points_idx, :] if self.use_multiple_poses else None,
            multiple_action_nocs_gt=multiple_grasp_points_nocs if self.use_multiple_poses else None,
        )
        
        if self.use_minsnr_reweight:
            minsnr_weights = self.diffusion_head.get_minsnr_weights(timesteps)
        else:
            minsnr_weights = None
        if self.diffusion_head.noise_scheduler.config.prediction_type == 'sample':
            loss_diffusion, loss_diffusion_weighted = self.sym_diffusion_mse_loss(fling_action_gt, fling_action_pred, weights=weights, minsnr_weights=minsnr_weights, use_sym_loss=self.use_sym_loss)
        else:
            loss_diffusion, loss_diffusion_weighted = self.epsilon_diffusion_mse_loss(fling_action_gt, fling_action_pred, weights=weights, minsnr_weights=minsnr_weights)
        loss_diffusion *= self.loss_diffusion_weight
        loss_diffusion_weighted *= self.loss_diffusion_weight
        if self.use_minsnr_reweight:
            loss_diffusion = loss_diffusion.detach()
        else:
            loss_diffusion_weighted = loss_diffusion_weighted.detach()

        err_grasp_variety_nocs, err_grasp_variety_xyz = \
            self.sym_grasp_mse_variety_cls_err(fling_action_nocs_pred,
                                                  pc_nocs if use_gt_nocs else pred_nocs,
                                                  pc_xyz,
                                                  gripper_points[:, self.gripper_points_idx, :],
                                                  grasp_points_nocs,
                                                  weights=weights,
                                                  use_xyz_variety_loss=self.state_head.use_xyz_variety_loss,
                                                  alpha=self.state_head.nocs_distance_weight_alpha)

        # nocs loss
        loss_nocs = self.loss_nocs_weight * self.sym_nocs_huber_loss(pred_nocs, pc_nocs, self.rescale_nocs)

        # smoothed loss and metrics
        loss_smoothed, smoothed_accuracy, smoothed_false_positive, smoothed_false_negative, smoothed_auc = \
            self._calculate_smoothed_loss_and_metrics(pred_smoothed_logits, smoothed_label)
        
        # smoothing style loss and metrics
        # loss_smoothing_style, smoothing_style_accuracy = \
        #     self._calculate_smoothing_style_loss_and_metrics(pred_smoothing_style_logits, smoothing_style_label)
        
        # keypoint detection loss and metrics
        loss_keypoint_mse, keypoint_err = \
            self._calculate_keypoint_detection_loss_and_metrics(pred_keypoints, keypoints, smoothed_label)

        # summarize loss and errors
        loss_dict = dict()
        loss_dict.update({'loss_diffusion': loss_diffusion,
                          'loss_diffusion_weighted': loss_diffusion_weighted,
                          'err_fling_grasp_xyz': err_grasp_variety_xyz,
                          'err_fling_grasp_nocs': err_grasp_variety_nocs,
                          'loss_nocs': loss_nocs,
                          'loss_smoothed': loss_smoothed,
                          'smoothed_accuracy': smoothed_accuracy,
                          'smoothed_false_positive': smoothed_false_positive,
                          'smoothed_false_negative': smoothed_false_negative,
                          'smoothed_auc': smoothed_auc,
                        #   'loss_smoothing_style': loss_smoothing_style,
                        #   'smoothing_style_accuracy': smoothing_style_accuracy,
                          'loss_keypoint_mse': loss_keypoint_mse,
                          'keypoint_err': keypoint_err,
                          'lr': self.optimizers().optimizer.param_groups[0]['lr'],
                          })
        loss_all = torch.zeros_like(loss_dict['loss_nocs'])
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item
        loss_dict['supervised_loss'] = loss_all
        return loss_dict
    
    def forward_supervised_classification_detection_loss(self, input_batch: tuple):
        coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, keypoints, rotation_cls, \
            reward, primitive_index_batch, smoothed_label, smoothing_style_label, folding_score, folding_step = \
            tuple(x.to(self.device) for x in input_batch)
        if self.use_multiple_poses:
            multiple_gripper_points = gripper_points.clone()
            multiple_grasp_points_nocs = grasp_points_nocs.clone()
            chosen_pose_idx = np.random.randint(0, gripper_points.shape[1])
            gripper_points = multiple_gripper_points[:, chosen_pose_idx]
            grasp_points_nocs = multiple_grasp_points_nocs[:, chosen_pose_idx]
            rotation_cls = rotation_cls[:, chosen_pose_idx]
        # network forward
        dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, pred_smoothed_logits, pred_smoothing_style_logits, pred_keypoints = \
            self.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)

        # nocs loss
        loss_nocs = self.loss_nocs_weight * self.sym_nocs_huber_loss(pred_nocs, pc_nocs, self.rescale_nocs)

        # smoothed loss and metrics
        loss_smoothed, smoothed_accuracy, smoothed_false_positive, smoothed_false_negative, smoothed_auc = \
            self._calculate_smoothed_loss_and_metrics(pred_smoothed_logits, smoothed_label)
        
        # smoothing style loss and metrics
        # loss_smoothing_style, smoothing_style_accuracy = \
        #     self._calculate_smoothing_style_loss_and_metrics(pred_smoothing_style_logits, smoothing_style_label)
        
        # keypoint detection loss and metrics
        loss_keypoint_mse, keypoint_err = \
            self._calculate_keypoint_detection_loss_and_metrics(pred_keypoints, keypoints, smoothed_label)

        # summarize loss and errors
        loss_dict = dict()
        loss_dict.update({'loss_nocs': loss_nocs,
                          'loss_smoothed': loss_smoothed,
                          'smoothed_accuracy': smoothed_accuracy,
                          'smoothed_false_positive': smoothed_false_positive,
                          'smoothed_false_negative': smoothed_false_negative,
                          'smoothed_auc': smoothed_auc,
                        #   'loss_smoothing_style': loss_smoothing_style,
                        #   'smoothing_style_accuracy': smoothing_style_accuracy,
                          'loss_keypoint_mse': loss_keypoint_mse,
                          'keypoint_err': keypoint_err,
                          'lr': self.optimizers().optimizer.param_groups[0]['lr'],
                          })
        loss_all = torch.zeros_like(loss_dict['loss_nocs'])
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item
        loss_dict['supervised_loss'] = loss_all
        return loss_dict
    

    def predict(self, pc_xyz_batch: torch.Tensor,
                coords: torch.Tensor,
                feat: torch.Tensor,
                action_type: ActionTypeDef,
                return_timing=False):
        timing = {}
        start = time.time()
        if not torch.is_tensor(coords) and not torch.is_tensor(feat):
            coords = torch.from_numpy(coords).to(device=self.device)
            feat = torch.from_numpy(feat).to(device=self.device)

        pre_processing = time.time()
        timing['pre_processing'] = pre_processing - start

        # use reference_model to predict points
        if self.reference_model is not None:
            model = self.reference_model
        else:
            model = self
        # network forward
        dense_feat_att, pred_nocs, use_gt_nocs, dense_nocs_feat_fuse, smoothed_logits, smoothing_style_logits, keypoints = \
            model.forward(coords, feat, pc_xyz_batch)

        nn_inference = time.time()
        timing['nn_inference'] = nn_inference - pre_processing

        assert action_type is not None, 'action type must not be NONE！'
        action_str = ActionTypeDef.to_string(action_type)
        action_idx = ActionTypeDef.from_string(action_str).value

        nocs_map = pred_nocs[0].detach().cpu().numpy()  # (N, 3)

        grasp_point_all = np.zeros((0, 3)).astype(np.float32)
        grasp_point_nocs_all = np.zeros((0, 3)).astype(np.float32)
        virtual_reward_all = np.zeros((0, 0, 1)).astype(np.float32)
        real_reward_all = np.zeros((0, 0, 1)).astype(np.float32)

        # always handle fling predictions for whatever action type
        num_pts = pred_nocs.shape[1]  # N
        num_candidates = model.state_head.num_pred_fling_candidates  # K

        action_xyz, action_nocs, action_xyz_list = model.diffusion_head.conditional_sample(
            context=dense_nocs_feat_fuse,
            pc_xyz=pc_xyz_batch,
            pred_pc_nocs=pred_nocs
        )

        pred_grasp_point_torch = action_xyz[0]  # (K, 3)

        grasp_point_all = pred_grasp_point_torch.detach().cpu().numpy()  # (K, 3)
        release_point_all = np.zeros_like(grasp_point_all)  # (K, 3)
        pred_grasp_point_nocs = action_nocs[0]  # (K, 3)
        grasp_point_nocs_all = pred_grasp_point_nocs.detach().cpu().numpy()  # (K, 3)

        if self.use_dpo_reward_for_inference:
            pair_score_all = torch.ones(num_candidates, num_candidates, device=self.device) * -np.inf  # (K, K)
            virtual_reward_all = pair_score_all.unsqueeze(-1)  # (K, K, 1)
            pred_grasp_xyz = action_xyz.repeat(self.dpo_reward_sample_num, 1, 1) # (N, K, 3)
            pred_grasp_nocs = action_nocs.repeat(self.dpo_reward_sample_num, 1, 1) # (N, K, 3)
            pc_xyz = pc_xyz_batch.repeat(self.dpo_reward_sample_num, 1, 1)  # (N, Np, 3)
            pc_nocs = pred_nocs.repeat(self.dpo_reward_sample_num, 1, 1)  # (N, Np, 3)
            timesteps = torch.randint(
                0,
                self.diffusion_head.noise_scheduler.config.num_train_timesteps // 10, # use top 10% timesteps
                (pred_grasp_xyz.shape[0],),
                device=coords.device
            ).long()
            fling_action_gt, fling_action_pred, timesteps, noise = self.get_diffusion_prediction_from_model(
                self, coords, feat, pc_xyz, pc_nocs, pred_grasp_xyz, pred_grasp_nocs, given_timesteps=timesteps
            )
            _, fling_action_pred_reference, timesteps, noise = self.get_diffusion_prediction_from_model(
                self.reference_model, coords, feat, pc_xyz, pc_nocs, pred_grasp_xyz, pred_grasp_nocs, given_timesteps=timesteps, given_noise=noise
            )
            if self.use_minsnr_reweight:
                minsnr_weights = self.diffusion_head.get_minsnr_weights(timesteps)
                minsnr_weights = minsnr_weights.unsqueeze(1)
            else:
                minsnr_weights = torch.tensor(1.0)
            
            B, K, D = fling_action_gt.shape
            fling_action_gt = fling_action_gt.reshape(B, K//2, 2, D)
            fling_action_pred = fling_action_pred.reshape(B, K//2, 2, D)
            fling_action_pred_reference = fling_action_pred_reference.reshape(B, K//2, 2, D)
            
            metric = nn.MSELoss(reduction='none')
            elbo = metric(fling_action_gt, fling_action_pred).mean(dim=(2, 3))
            elbo_reference = metric(fling_action_gt, fling_action_pred_reference).mean(dim=(2, 3))
            reward = -self.dpo_beta * minsnr_weights * (elbo - elbo_reference)
            reward = reward.mean(dim=0)
            for k in range(num_candidates // 2):
                pair_score_all[2*k, 2*k + 1] = reward[k]
            logger.info("Use DPO reward for inference: {}".format(reward))
            # print(pair_score_all)
        elif self.use_reward_prediction_for_inference:
            pair_score_all = torch.ones(num_candidates, num_candidates, device=self.device) * -np.inf  # (K, K)
            virtual_reward_all = pair_score_all.unsqueeze(-1)  # (K, K, 1)

            action_xyz_gt = action_xyz.clone()
            _, _, _, dense_nocs_feat_fuse_self, _, _, _ = \
                self.forward(coords, feat, pc_xyz_batch)
            reward = self.reward_head(action_xyz_gt, dense_nocs_feat_fuse_self, pc_xyz_batch)
            reward1, reward2 = torch.chunk(reward, 2, dim=-1)
            print(reward.shape, reward1.shape, reward2.shape)
            reward = torch.cat([reward1, reward2], dim=1).reshape(-1)

            for k in range(num_candidates // 2):
                pair_score_all[2 * k, 2 * k + 1] = reward[k]
            logger.info("Use reward prediction for inference: {}".format(reward))
        else:
            pair_score_all = torch.zeros(num_candidates, num_candidates, device=self.device)  # (K, K)
            virtual_reward_all = pair_score_all.unsqueeze(-1)  # (K, K, 1)
        # TODO: implement this
        real_reward_all = torch.zeros_like(virtual_reward_all)
        # make sure that the same point is not selected
        self_idxs = np.arange(num_candidates)
        pair_score_all[self_idxs, self_idxs] = -np.inf
        if self.random_select_diffusion_action_pair_for_inference:
            for k in range(num_candidates // 2):
                pair_score_all[2*k, 2*k + 1] = 100.0
        # make sure that a pair of points is only selected once
        # for i in range(num_candidates):
        #     for j in range(num_candidates):
        #         if i < j:
        #             pair_score_all[i, j] = 0.
        flatten_pair_score = pair_score_all.view(-1)  # (K*K, )
        flatten_pair_score_numpy = flatten_pair_score.detach().cpu().numpy()  # (K*K, )
        sorted_pair_idxs = np.argsort(flatten_pair_score_numpy)[::-1]  # (K*K, )
        virtual_reward_all = virtual_reward_all.detach().cpu().numpy()  # (K, K, 1)
        real_reward_all = real_reward_all.cpu().numpy()  # (K, K, 1)

        def action_iterator(random: bool = False, random_top_ratio: float = 1.0, top_ratio: float = 1.0) -> ActionIteratorMessage:
            """iterate valid pose pairs (sorted by scores) from possible samples"""
            nonlocal sorted_pair_idxs
            if action_str == 'fling' or action_str == 'sweep' or action_str == 'single_pick_and_place':
                if random:
                    # only for self-supervised data collection
                    # randomly shuffle sorted_pair_idxs
                    # only shuffle top K pairs
                    top_k = int(len(sorted_pair_idxs) * random_top_ratio)
                    np.random.shuffle(sorted_pair_idxs[:top_k])
                else:
                    top_k = int(len(sorted_pair_idxs) * top_ratio)
                if self.manually_select_diffusion_action_pair_for_inference:
                    selected_idx = py_cli_interaction.must_parse_cli_sel("select the most suitable action pair", list(range(top_k)))
                    selected_pair_idx = None
                    for idx, pair_idx in enumerate(sorted_pair_idxs[:top_k]):
                        idx1 = pair_idx // num_candidates
                        idx2 = pair_idx % num_candidates
                        if idx1 // 2 == selected_idx or idx2 // 2 == selected_idx:
                            selected_pair_idx = idx
                    sorted_pair_idxs = np.concatenate([
                        sorted_pair_idxs[[selected_pair_idx]],
                        sorted_pair_idxs[:selected_pair_idx],
                        sorted_pair_idxs[selected_pair_idx + 1:]
                    ])
                for idx in range(top_k):
                    pair_idx = sorted_pair_idxs[idx]
                    idx1 = pair_idx // num_candidates
                    idx2 = pair_idx % num_candidates
                    if action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                        poses = np.stack([grasp_point_all[idx1], grasp_point_all[idx1],
                                          grasp_point_all[idx2], grasp_point_all[idx2]], axis=0)
                    else:
                        poses = np.stack([grasp_point_all[idx1], grasp_point_all[idx2],
                                        release_point_all[idx1], release_point_all[idx2]], axis=0)  # (34 3)
                    # TODO: add predicted rotation angles in poses
                    poses = np.concatenate([poses, np.zeros((4, 1))], axis=1)  # (x, y, z, rot)
                    poses_nocs = np.stack([grasp_point_nocs_all[idx1], grasp_point_nocs_all[idx2]], axis=0)  # (2, 3)
                    msg = ActionIteratorMessage(poses_4d=poses, poses_nocs=poses_nocs, grasp_idxs=(int(idx1), int(idx2)))
                    yield msg
                return None
            else:
                raise NotImplementedError

        pred_message = PredictionMessage(action_type=action_type,
                                         action_iterator=action_iterator,
                                         action_xyz_diffusion=action_xyz[0].detach().cpu().numpy(),
                                         action_nocs_diffusion=action_nocs[0].detach().cpu().numpy(),
                                         action_xyz_list_diffusion=[action_xyz[0].detach().cpu().numpy() for action_xyz in action_xyz_list],
                                         nocs_map=nocs_map,
                                         grasp_point_all=grasp_point_all,
                                         grasp_point_nocs_all=grasp_point_nocs_all,
                                         virtual_reward_all=virtual_reward_all,
                                         real_reward_all=real_reward_all)
        if return_timing:
            pred_message.nn_timing = timing
        return pred_message

    @staticmethod
    def save_vis_single(pc_xyz, attmaps, pc_nocs, save_dir, pred_action, pred_keypoints, pred_grasp_nocs, pcd_id=0,
                        pred_keypoints_all=None, pred_grasp_nocs_all=None):
        if pred_action == 'fling':
            attmap = attmaps['fling_pick'].T  # (K, N)
        elif pred_action == 'drag':
            attmap = np.stack(
                [attmaps['drag_pick1'], attmaps['drag_pick2'], attmaps['drag_place1'], attmaps['drag_place2']])
        elif pred_action == 'fold1':
            attmap = np.stack(
                [attmaps['fold1_pick1'], attmaps['fold1_pick2'], attmaps['fold1_place1'], attmaps['fold1_place2']])
        elif pred_action == 'fold2':
            attmap = np.stack(
                [attmaps['fold2_pick1'], attmaps['fold2_pick2'], attmaps['fold2_place1'], attmaps['fold2_place2']])
        elif pred_action == 'pick_and_place':
            attmap = np.stack(
                [attmaps['pnp_pick1'], attmaps['pnp_pick2'], attmaps['pnp_place1'], attmaps['pnp_place2']])
        else:
            attmap = np.stack(
                [attmaps['fold1_pick1'], attmaps['fold1_pick2'], attmaps['fold1_place1'], attmaps['fold1_place2']])
        data_dict = dict(pc_xyz=pc_xyz.tolist(),
                         pc_nocs=pc_nocs.tolist(),
                         attmaps=attmap.tolist(),
                         grasp_nocs=pred_grasp_nocs.tolist(),
                         action=pred_action,
                         pred_keypoints_all=pred_keypoints_all.tolist() if pred_keypoints_all is not None else None,
                         pred_grasp_nocs_all=pred_grasp_nocs_all.tolist() if pred_grasp_nocs_all is not None else None,
                         left_gripper_point_frame1=pred_keypoints[0, :3].tolist(),
                         right_gripper_point_frame1=pred_keypoints[1, :3].tolist(),
                         left_gripper_point_frame2=pred_keypoints[2, :3].tolist(),
                         right_gripper_point_frame2=pred_keypoints[3, :3].tolist(),
                         left_theta_frame1=pred_keypoints[0, 3].tolist(),
                         right_theta_frame1=pred_keypoints[1, 3].tolist(),
                         left_theta_frame2=pred_keypoints[2, 3].tolist(),
                         right_theta_frame2=pred_keypoints[3, 3].tolist())
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
        data = bson.BSON.encode(data_dict)
        out_path = osp.join(save_dir, 'vis', '{:0>4d}.bson'.format(pcd_id))
        with open(out_path, 'wb') as f:
            f.write(data)
        print('Saving to {}! Action type: {}'.format(out_path, pred_action))
        
    def infer_finetune(self, batch, batch_idx, is_train=True):
        for idx, data in enumerate(batch):
            if isinstance(data, torch.Tensor) and torch.any(~torch.isfinite(data)):
                logger.error(f'supervised data has NaN or Inf: idx {idx}, gt data {data}!')
                raise RuntimeError
        metrics = self.forward_finetune_loss(batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value, sync_dist=True)
        return metrics

    def infer_supervised(self, batch, batch_idx, is_train=True):
        for idx, data in enumerate(batch):
            if isinstance(data, torch.Tensor) and torch.any(~torch.isfinite(data)):
                logger.error(f'supervised data has NaN or Inf: idx {idx}, gt data {data}!')
                raise RuntimeError
        if self.enable_new_pipeline_supervised_classification_detection:
            metrics = self.forward_supervised_classification_detection_loss(batch)
        else:
            metrics = self.forward_supervised_loss(batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value, sync_dist=True)
        return metrics

    def training_step(self, batch, batch_idx):
        if len(batch) == 2 and isinstance(batch[0], tuple) and isinstance(batch[1], tuple):
            # Use min_size Combiner for training with multiple dataloaders.
            # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
            metrics_supervised = self.infer_supervised(batch[0], batch_idx, is_train=True)
            metrics_extra = self.infer_extra(batch[1], batch_idx, is_train=True)
            metrics = {**metrics_supervised, **metrics_extra}
            metrics['loss'] = metrics_supervised['supervised_loss'] + metrics_extra['extra_loss']
        else:
            if self.enable_new_pipeline_finetune:
                metrics = self.infer_finetune(batch, batch_idx, is_train=True)
                metrics['loss'] = metrics['finetune_loss']
            else:
                metrics = self.infer_supervised(batch, batch_idx, is_train=True)
                metrics['loss'] = metrics['supervised_loss']
        if torch.any(~torch.isfinite(metrics['loss'])):
            metrics['loss'] = torch.zeros_like(metrics['loss'], requires_grad=True)
            logger.warning('NaN or Inf detected in loss. Skipping this batch.')
        self.log('train_loss', metrics['loss'], sync_dist=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # Use SequentialCombiner for validation.
        # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
        if dataloader_idx == 0 or dataloader_idx is None:
            if self.enable_new_pipeline_finetune:
                metrics = self.infer_finetune(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['finetune_loss']
            else:
                metrics = self.infer_supervised(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['supervised_loss']
        elif dataloader_idx == 1:
            metrics = self.infer_extra(batch, batch_idx, is_train=False)
            metrics['loss'] = metrics['extra_loss']
        else:
            raise NotImplementedError
        self.log('val_loss', metrics['loss'], sync_dist=True)
        return metrics['loss']
