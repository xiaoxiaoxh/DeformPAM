import sys
import os
import os.path as osp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl

from common.datamodels import ActionTypeDef
from learning.datasets.imitation_dataset5 import ImitationDataModule5
from learning.datasets.runtime_dataset_real import RuntimeDataModuleReal
from learning.net.primitive_diffusion import PrimitiveDiffusion

from tqdm import tqdm
from bson import Binary
import pickle


def calculate_variety_loss(action_xyz_gt: torch.Tensor,
                           action_nocs_gt: torch.Tensor,
                           action_xyz_pred: torch.Tensor,
                           action_nocs_pred: torch.Tensor,
                           weights: float = 1.0):
    K = action_nocs_pred.shape[0]

    nocs_metric = torch.nn.MSELoss(reduction='none')
    left_target_nocs = action_nocs_gt[0, :].unsqueeze(0).expand(K, -1)  # (K, 3)
    right_target_nocs = action_nocs_gt[1, :].unsqueeze(0).expand(K, -1)  # (K, 3)

    left_grasp_loss_nocs = nocs_metric(action_nocs_pred, left_target_nocs).mean(dim=-1)  # (K)
    right_grasp_loss_nocs = nocs_metric(action_nocs_pred, right_target_nocs).mean(dim=-1)  # (K)

    left_variety_loss_nocs, left_target_idxs_nocs = torch.min(left_grasp_loss_nocs, dim=0)  # (,)
    right_variety_loss_nocs, right_target_idxs_nocs = torch.min(right_grasp_loss_nocs, dim=0)  # (,)
    variety_loss_nocs  = (left_variety_loss_nocs + right_variety_loss_nocs) / 2.0
    error_nocs = (variety_loss_nocs * weights).sum()

    xyz_metric = torch.nn.MSELoss(reduction='none')
    left_target_xyz = action_xyz_gt[0, :].unsqueeze(0).expand(K, -1)  # (K, 3)
    right_target_xyz = action_xyz_gt[1, :].unsqueeze(0).expand(K, -1)  # (K, 3)

    left_grasp_loss_xyz = xyz_metric(action_xyz_pred, left_target_xyz).mean(dim=-1)  # (K)
    right_grasp_loss_xyz = xyz_metric(action_xyz_pred, right_target_xyz).mean(dim=-1)  # (K)
    left_variety_loss_xyz = left_grasp_loss_xyz[left_target_idxs_nocs]  # (,)
    right_variety_loss_xyz = right_grasp_loss_xyz[right_target_idxs_nocs]  # (,)
    variety_loss_xyz = (left_variety_loss_xyz + right_variety_loss_xyz) / 2.0  # (,)
    error = (variety_loss_xyz * weights).sum()

    return error, error_nocs

# %%
# main script
@hydra.main(config_path="output", config_name="config", version_base='1.1')
def main(cfg: DictConfig) -> None:
    test_model_dir = cfg.output_dir
    test_model_name = cfg.config.logger.run_name
    output_dir = os.getcwd()
    save_bson = cfg.save_bson
    scheduler_type = cfg.scheduler_type
    num_inference_steps = cfg.num_inference_steps
    ddim_eta = cfg.ddim_eta
    cfg = cfg.config
    
    model_params = {**cfg.model}
    model_params['diffusion_head_params'] = {**model_params['diffusion_head_params']}
    model_params['diffusion_head_params']['scheduler_type'] = scheduler_type
    model_params['diffusion_head_params']['num_inference_steps'] = num_inference_steps
    model_params['diffusion_head_params']['ddim_eta'] = ddim_eta
    model = PrimitiveDiffusion(**model_params)
    checkpoint_path = osp.join(test_model_dir, "checkpoints")
    for file in os.listdir(checkpoint_path):
        if 'last' in file:
            checkpoint_path = osp.join(checkpoint_path, file)
            break
    state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    if cfg.runtime_datamodule is None:
        cfg.datamodule.batch_size = 1
        datamodule = ImitationDataModule5(**cfg.datamodule)
        datamodule.prepare_data()
    else:
        cfg.runtime_datamodule.batch_size = 1
        datamodule = RuntimeDataModuleReal(**cfg.runtime_datamodule)
        datamodule.prepare_data()
        datamodule.test_subset.dataset.use_augmentation = True # use augmentation to normalize the input
        if datamodule.test_subset.dataset.data_type == 'new_pipeline_finetune': # only one gt pose in finetune dataset
            datamodule.test_subset.dataset.data_type = 'new_pipeline_supervised'
            datamodule.test_subset.dataset.return_multiple_poses = False
            cfg.runtime_datamodule.return_multiple_poses = False
            model.use_multiple_poses = False

    inference_idx = -1
    errors, errors_nocs = [], []
    loop = tqdm(datamodule.test_dataloader())
    for data_batch in loop:
        coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, rotation_cls, \
            reward, primitive_index_batch, smoothed_score, folding_score, folding_step = data_batch
        
        if cfg.runtime_datamodule is not None and getattr(cfg.runtime_datamodule, "return_multiple_poses", False):
            multiple_gripper_points = gripper_points.clone()
            multiple_grasp_points_nocs = grasp_points_nocs.clone()
            chosen_pose_idx = np.random.randint(0, gripper_points.shape[1])
            gripper_points = multiple_gripper_points[:, chosen_pose_idx]
            grasp_points_nocs = multiple_grasp_points_nocs[:, chosen_pose_idx]
            rotation_cls = rotation_cls[:, chosen_pose_idx]
        else:
            multiple_gripper_points = None
            multiple_grasp_points_nocs = None
        
        inference_idx += 1

        if primitive_index_batch[0] != 0:
            print("skip")
            continue
        
        PredictionMessage = model.predict(pc_xyz, coords, feat, action_type=ActionTypeDef.FLING)

        # save pc, action_xyz_diffusion, action_xyz_list_diffusion as bson file
        data_dict = dict()
        data_dict[f'pc_xyz'] = pc_xyz[0].cpu().numpy().tolist()
        data_dict[f'action_xyz_list_diffusion'] = PredictionMessage.action_xyz_list_diffusion
        action_xyz_pred = data_dict[f'action_xyz_diffusion'] = PredictionMessage.action_xyz_diffusion
        action_nocs_pred = data_dict[f'action_nocs_diffusion'] = PredictionMessage.action_xyz_diffusion
        action_xyz_pred = torch.from_numpy(action_xyz_pred)
        action_nocs_pred = torch.from_numpy(action_nocs_pred)

        action_xyz_gt = gripper_points[0, 0:3:2, :]  # (2, 3)
        data_dict[f'action_xyz_gt'] = action_xyz_gt.cpu().numpy().tolist()
        action_nocs_gt = grasp_points_nocs[0]  # (2, 2)
        data_dict[f'action_nocs_gt'] = action_nocs_gt.cpu().numpy().tolist()
        if multiple_gripper_points is not None:
            multiple_action_xyz_gt = multiple_gripper_points[0, :, 0:3:2, :]
            data_dict[f'multiple_action_xyz_gt'] = multiple_action_xyz_gt.cpu().numpy().tolist()
        if multiple_grasp_points_nocs is not None:
            multiple_action_nocs_gt = multiple_grasp_points_nocs[0, :, :]
            data_dict[f'multiple_action_nocs_gt'] = multiple_action_nocs_gt.cpu().numpy().tolist()

        error, error_nocs = calculate_variety_loss(action_xyz_gt, action_nocs_gt, action_xyz_pred, action_nocs_pred)

        errors.append(error.item())
        errors_nocs.append(error_nocs.item())
        loop.set_postfix(mean_error=np.mean(errors), mean_error_nocs=np.mean(errors_nocs))
        loop.refresh()

        if save_bson:
            with open(os.path.join(output_dir, f'{test_model_name}_{inference_idx}.bson'), 'wb') as f:
                f.write(Binary(pickle.dumps(data_dict, protocol=2), subtype=128))


# %%
# driver
if __name__ == "__main__":
    main()
