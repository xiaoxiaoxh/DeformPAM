import sys
import os
import os.path as osp
import psutil

import py_cli_interaction
from rich.console import Console

sys.path.append(osp.join('', os.path.dirname(os.path.abspath(__file__))))

import time
import open3d as o3d
from loguru import logger
from typing import Dict, Tuple, Optional, Union, List, Any
from autolab_core import RigidTransform
from learning.net.primitive_diffusion import PrimitiveDiffusion
from common.space_util import depth_to_point, point_to_depth_map_location
from common.datamodels import (ActionTypeDef, ObjectTypeDef, ActionMessage, ObservationMessage, ExceptionMessage,
                               PredictionMessage, GeneralObjectState, GarmentSmoothingStyle, ObjectState, ActionIteratorMessage)
from common.experiment_base import ExperimentBase
from manipulation.experiment_real import ExperimentReal
from common.visualization_util import visualize_pc_and_grasp_points
from common.pcd_utils import FPS, pick_n_points_from_pcd
from common.space_util import transform_point_cloud
from manipulation.executable_checker import setJointSpaceState

import numpy as np
import torch
import cv2
import math
import MinkowskiEngine as ME

from omegaconf import OmegaConf
from omegaconf import DictConfig

from tools.remote_operation.server import remote_pick_n_points_from_pcd
from functools import partial
import itertools
import random
from copy import deepcopy
from multiprocessing import Process, Queue


class Inference3D:
    """
    Inference class for 3D point cloud input
    use action iterator for network prediction
    """
    __VERSION__ = "v3"

    def __init__(
            self,
            model_path: str,
            classification_detection_model_path: Optional[str],
            model_name: str = 'last',
            model_version: str = 'v7',
            experiment: Union[ExperimentReal, ExperimentBase] = None,  # Experiment class
            args: Union[OmegaConf, DictConfig] = None,
            **kwargs):
        self.experiment = experiment
        if args.use_parallel_checker:
            self.checker_task_queue = Queue()
            self.checker_result_queue = Queue()
            self.checker_workers = [Process(target=self._search_pick_p_y_fixer_worker,
                                            args=(self.experiment.option, self.checker_task_queue, self.checker_result_queue))
                                    for _ in range(args.checker_processes_num)]
            for worker in self.checker_workers:
                worker.start()
        # load model to gpu
        assert model_version in ('v4', 'v5', 'v6', 'v7', 'diffusion_v1'), f'model version {model_version} does not exist!'

        checkpoint_dir = osp.join(model_path, 'checkpoints')
        checkpoint_path = osp.join(checkpoint_dir, model_name + '.ckpt')
        if classification_detection_model_path is not None:
            classification_detection_checkpoint_dir = osp.join(classification_detection_model_path, 'checkpoints')
            classification_detection_checkpoint_path = osp.join(classification_detection_checkpoint_dir, model_name + '.ckpt')
        model_config = OmegaConf.load(osp.join(model_path, 'config.yaml'))
        # data hyper-params
        if 'diffusion' in model_version:
            self.voxel_size: float = model_config.config.runtime_datamodule.voxel_size
        else:
            self.voxel_size: float = model_config.config.datamodule.voxel_size
        self.num_pc_sample: int = model_config.config.datamodule.num_pc_sample
        self.num_pc_sample_final: int = model_config.config.datamodule.num_pc_sample_final

        logger.info(f'loading model from {checkpoint_path}!')
        if model_version == 'diffusion_v1':
            model_cpu = PrimitiveDiffusion.load_from_checkpoint(checkpoint_path, strict=False)
            if classification_detection_model_path is not None:
                classification_detection_model_cpu = PrimitiveDiffusion.load_from_checkpoint(
                    classification_detection_checkpoint_path, strict=False)
        else:
            raise NotImplementedError
        self.model_version = model_version
        
        if 'diffusion' in model_version and hasattr(args, 'model'):
            # update model config
            model_cpu.use_virtual_reward_for_inference = getattr(args.model, 'use_virtual_reward_for_inference',
                                                                    model_cpu.use_virtual_reward_for_inference)
            model_cpu.use_dpo_reward_for_inference = getattr(args.model, 'use_dpo_reward_for_inference', 
                                                             model_cpu.use_dpo_reward_for_inference)
            model_cpu.use_reward_prediction_for_inference = getattr(args.model, 'use_reward_prediction_for_inference',
                                                                    model_cpu.use_reward_prediction_for_inference)
            model_cpu.dpo_reward_sample_num = getattr(args.model, 'dpo_reward_sample_num',
                                                        model_cpu.dpo_reward_sample_num)
            model_cpu.random_select_diffusion_action_pair_for_inference = getattr(args.model, 'random_select_diffusion_action_pair_for_inference',
                                                                                    model_cpu.random_select_diffusion_action_pair_for_inference)
            model_cpu.manually_select_diffusion_action_pair_for_inference = getattr(args.model, 'manually_select_diffusion_action_pair_for_inference',
                                                                                    model_cpu.manually_select_diffusion_action_pair_for_inference)
            if model_cpu.manually_select_diffusion_action_pair_for_inference:
                if not model_cpu.random_select_diffusion_action_pair_for_inference:
                    logger.warning("manually select action pair is not enabled when randomly select action pair is disabled")
                    model_cpu.manually_select_diffusion_action_pair_for_inference = False

            inference_point_num = getattr(args.model, 'inference_point_num', model_cpu.state_head.num_pred_candidates)
            model_cpu.state_head.num_pred_candidates = inference_point_num
            model_cpu.diffusion_head.num_of_grasp_points = inference_point_num

        if 'diffusion' in model_version and hasattr(args, 'model') and hasattr(args.model, 'diffusion_head_params'):
            # update model config
            model_cpu.diffusion_head.scheduler_type = args.model.diffusion_head_params.scheduler_type
            model_cpu.diffusion_head.num_inference_steps = args.model.diffusion_head_params.num_inference_steps
            model_cpu.diffusion_head.ddim_eta = args.model.diffusion_head_params.ddim_eta
        
        if model_cpu.reference_model is not None:
            model_cpu.sync_reference_model_settings()
               
        # TODO: use fixed seed
        device = torch.device('cuda:0')
        self.model = model_cpu.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        if classification_detection_model_path is not None:
            self.classification_detection_model = classification_detection_model_cpu.to(device)
            self.classification_detection_model.eval()
            self.classification_detection_model.requires_grad_(False)
        else:
            self.classification_detection_model = None

        # resolve config
        resolved_config = OmegaConf.to_container(args, resolve=True)
        resolved_config = OmegaConf.create(resolved_config)
        self.args = resolved_config

        # save center
        self.center = None
        # save scale
        self.scale = getattr(model_config.config, 'xy_normalize_factor', 1.0) 
        self.use_ood_points_removal = getattr(model_config.config.runtime_datamodule, 'use_ood_points_removal', False)
        
        self.test_info = {
            'success': 0,
            'failed': 0,
            'currrent_steps': 0,
            'history_steps': []
        }

    def __del__(self):
        if self.args.use_parallel_checker:
            for worker in self.checker_workers:
                self.checker_task_queue.put(None)
            for worker in self.checker_workers:
                worker.join()

    def remove_ood_points(self, pts_xyz: np.ndarray) -> np.ndarray:
        if self.use_ood_points_removal:
            pts_xyz_torch = torch.from_numpy(pts_xyz)
            # remove out-of-distribution points according to the mean and std
            mean = torch.mean(pts_xyz_torch, dim=0)
            std = torch.std(pts_xyz_torch, dim=0)
            in_distribution_idx = (pts_xyz_torch[:, 0] > mean[0] - 3 * std[0]) & (pts_xyz_torch[:, 0] < mean[0] + 3 * std[0]) & \
                                    (pts_xyz_torch[:, 1] > mean[1] - 3 * std[1]) & (pts_xyz_torch[:, 1] < mean[1] + 3 * std[1]) & \
                                    (pts_xyz_torch[:, 2] > mean[2] - 3 * std[2]) & (pts_xyz_torch[:, 2] < mean[2] + 3 * std[2])
            pts_xyz = pts_xyz_torch[in_distribution_idx].numpy()
        return pts_xyz
    
    def conduct_zero_center_and_scale(self, pts_xyz: np.ndarray) -> np.ndarray:
        if self.args.use_zero_center:
            pts_xyz = pts_xyz.copy()
            self.center = ((np.max(pts_xyz, axis=0) + np.min(pts_xyz, axis=0)) / 2)[np.newaxis, :]
            pts_xyz = pts_xyz - self.center
            pts_xyz *=  self.scale
            # no zero-center for z axis
            pts_xyz[:, 2] += self.center[:, 2]
        return pts_xyz

    def undo_zero_center_and_scale(self, pts_xyz: np.ndarray) -> np.ndarray:
        if self.args.use_zero_center:
            pts_xyz = pts_xyz.copy()
            # no zero-center for z axis
            pts_xyz[:, 2] -= self.center[:, 2]
            pts_xyz /= self.scale
            pts_xyz = pts_xyz + self.center
        return pts_xyz

    def transform_input(self, pts_xyz: np.ndarray, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(pts_xyz.shape[0])
        # random select fixed number of points
        if all_idxs.shape[0] >= self.num_pc_sample:
            selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample, replace=False)
        else:
            np.random.seed(seed)
            np.random.shuffle(all_idxs)
            res_num = len(all_idxs) - self.num_pc_sample
            selected_idxs = np.concatenate([all_idxs, all_idxs[:res_num]], axis=0)
        pc_xyz_slim = pts_xyz[selected_idxs, :]

        # perform voxelization for Sparse ResUnet-3D
        _, sel_pc_idxs = ME.utils.sparse_quantize(pc_xyz_slim / self.voxel_size, return_index=True)
        origin_slim_pc_num = sel_pc_idxs.shape[0]
        assert origin_slim_pc_num >= self.num_pc_sample_final
        all_idxs = np.arange(origin_slim_pc_num)
        rs = np.random.RandomState(seed=seed)
        final_selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
        sel_pc_idxs = sel_pc_idxs[final_selected_idxs]
        assert sel_pc_idxs.shape[0] == self.num_pc_sample_final
        # voxelized coords for MinkowskiEngine engine
        coords = np.floor(pc_xyz_slim[sel_pc_idxs, :] / self.voxel_size)
        feat = pc_xyz_slim[sel_pc_idxs, :]
        # create sparse-tensor batch
        coords, feat = ME.utils.sparse_collate([coords], [feat])

        coords = coords.to(self.model.device)
        feat = feat.to(self.model.device)
        pts_xyz_batch = torch.from_numpy(pc_xyz_slim[sel_pc_idxs, :]).unsqueeze(0).to(self.model.device)
        return pts_xyz_batch, coords, feat

    def transform_output(self, 
                         poses: np.ndarray, 
                         pts_xyz: np.ndarray, 
                         action_type: ActionTypeDef,
                         pts_xyz_raw: np.ndarray = None,
                         fps_pick_override: bool = False,
                         nearest_pick_override: bool = False) -> \
            Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform]:
        """transform and fix output poses"""
        # poses definition:
        # 0: left pick
        # 1: right pick
        # 2: left place
        # 3: right place
        vertical_vector = None
        delta_length = getattr(self.experiment.option.compat.machine, "wiper_delta_length", 0.0)
        if action_type == ActionTypeDef.FLING:
            if fps_pick_override:
                logger.warning('Use FPS to find grasp points for fling action!')
                # predicted grasp points are too close, use FPS to randomly select grasp points
                fps = FPS(pts_xyz, 2)
                fps.fit()
                selected_pts = fps.get_selected_pts()
                poses[:2, :3] = selected_pts
            if nearest_pick_override:
                logger.warning('Use nearest point to find grasp points for fling action!')
                # predicted grasp points are too far from the object, use nearest point to find grasp points
                poses_xyz_expanded = np.expand_dims(poses[:2, :2], axis=1) # only use xy coordinates
                pts_xyz_expanded = np.expand_dims(pts_xyz[:, :2], axis=0)
                dist = np.linalg.norm(poses_xyz_expanded - pts_xyz_expanded, axis=-1)
                idxs = np.argmin(dist, axis=1)
                poses[:2, :3] = pts_xyz[idxs, :] # TODO: better implementation on z axis
            poses[:, -1] = -np.pi / 2
        elif action_type == ActionTypeDef.SWEEP:
            begin_offset = self.args.action_fixer.sweep.other_params.begin_offset
            end_offset = self.args.action_fixer.sweep.other_params.end_offset
            vertical_vector = poses[2, :2] - poses[0, :2]
            vertical_vector /= np.linalg.norm(vertical_vector)
            poses[:2, :2] -= begin_offset * vertical_vector
            poses[2:, :2] -= end_offset * vertical_vector
            theta = math.atan(vertical_vector[0] / (vertical_vector[1] + 1e-6))
            poses[:, -1] = np.pi + theta
        elif action_type == ActionTypeDef.SINGLE_PICK_AND_PLACE:
            if nearest_pick_override:
                logger.warning('Use nearest point to find grasp points for single pick and place action!')
                # predicted grasp points are too far from the rope, use nearest point to find grasp points
                poses_xyz_expanded = np.expand_dims(poses[:2, :2], axis=1)  # only use xy coordinates
                pts_xyz_expanded = np.expand_dims(pts_xyz[:, :2], axis=0)

                dist1 = np.linalg.norm(poses_xyz_expanded - pts_xyz_expanded, axis=-1)
                idx1 = np.argmin(dist1, axis=1)
                nearest_pt_xyz_expanded = np.expand_dims(pts_xyz[idx1, :2], axis=1)

                dist2 = np.linalg.norm(nearest_pt_xyz_expanded - pts_xyz_expanded, axis=-1)
                idxs2 = np.argsort(dist2, axis=1)[:, :200] # TODO: remove magic number
                poses[:2, :2] = np.mean(pts_xyz[idxs2, :2], axis=1)

        fixer_params = self.args.action_fixer.get(ActionTypeDef.to_string(action_type), None)
        assert fixer_params is not None, f"Action type {action_type} is not supported in action fixer!"
        poses6d = np.zeros((4, 6))
        poses6d[:, :3] = poses[:, :3]
        poses6d[:, 4] = poses[:, 3] # euler angle
        logger.debug(
            f"Raw action {action_type} in virtual space, poses_6d: {poses6d}")

        poses6d = self._predefined_fixer(pts_xyz_raw, poses6d, params=fixer_params.predefined_fixer_params)
        poses6d = self._edge_pick_fixer(pts_xyz_raw, poses6d, params=fixer_params.edge_pick_fixer_params)
        poses6d = self._search_pick_p_y_fixer(pts_xyz_raw, poses6d, action_type, vertical_vector, delta_length, params=fixer_params.search_pick_p_y_fixer_params)
        logger.debug(
            f"Fixed action {action_type} in virtual space, poses_6d: {poses6d}")

        euler_type = 'XZY' if action_type == ActionTypeDef.SWEEP else 'XZX'
        pick1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[0], euler_type=euler_type)
        pick2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[1], euler_type=euler_type)
        place1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[2], euler_type=euler_type)
        place2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[3], euler_type=euler_type)

        return pick1, pick2, place1, place2

    def _predefined_fixer(self,
                          pts_xyz_raw: np.ndarray,
                          poses: np.ndarray,
                          params: DictConfig) -> np.ndarray:
        """
        fix gripper translation based on predefined parameters
        input params: poses: (4, 6) ndarray
        # each row in poses: (x, y, z) + (r, p, y)
                # poses definition:
        # row 0: left pick
        # row 1: right pick
        # row 2: left place
        # row 3: right place
        """
        enable = params.enable
        if not enable:
            return poses

        if 'default' in params:
            # fix the default translation for all poses
            z = params.default.z
            r = math.radians(params.default.r) if params.default.r is not None else None
            p = math.radians(params.default.p) if params.default.p is not None else None
            y = math.radians(params.default.y) if params.default.y is not None else None

            poses[:, 2] = z
            poses[:, 3] = r if r is not None else poses[:, 3]
            poses[:, 4] = p if p is not None else poses[:, 4]
            poses[:, 5] = y if y is not None else poses[:, 5]

        if 'left' in params:
            # fix the left gripper pose
            z = params.left.z
            r = math.radians(params.left.r) if params.left.r is not None else None
            p = math.radians(params.left.p) if params.left.p is not None else None
            y = math.radians(params.left.y) if params.left.y is not None else None

            poses[0, 2] = z
            poses[0, 3] = r if r is not None else poses[0, 3]
            poses[0, 4] = p if p is not None else poses[0, 4]
            poses[0, 5] = y if y is not None else poses[0, 5]

            poses[2, 2] = z
            poses[2, 3] = r if r is not None else poses[2, 3]
            poses[2, 4] = p if p is not None else poses[2, 4]
            poses[2, 5] = y if y is not None else poses[2, 5]

        if 'right' in params:
            # fix the right gripper pose
            z = params.right.z
            r = math.radians(params.right.r) if params.right.r is not None else None
            p = math.radians(params.right.p) if params.right.p is not None else None
            y = math.radians(params.right.y) if params.right.y is not None else None

            poses[1, 2] = z
            poses[1, 3] = r if r is not None else poses[1, 3]
            poses[1, 4] = p if p is not None else poses[1, 4]
            poses[1, 5] = y if y is not None else poses[1, 5]

            poses[3, 2] = z
            poses[3, 3] = r if r is not None else poses[3, 3]
            poses[3, 4] = p if p is not None else poses[3, 4]
            poses[3, 5] = y if y is not None else poses[3, 5]

        if 'start_left' in params:
            # fix the left gripper start pose
            z = params.start_left.z
            r = math.radians(params.start_left.r) if params.start_left.r is not None else None
            p = math.radians(params.start_left.p) if params.start_left.p is not None else None
            y = math.radians(params.start_left.y) if params.start_left.y is not None else None
            poses[0, 2] = z
            poses[0, 3] = r if r is not None else poses[0, 3]
            poses[0, 4] = p if p is not None else poses[0, 4]
            poses[0, 5] = y if y is not None else poses[0, 5]

        if 'start_right' in params:
            # fix the right gripper start pose
            z = params.start_right.z
            r = math.radians(params.start_right.r) if params.start_right.r is not None else None
            p = math.radians(params.start_right.p) if params.start_right.p is not None else None
            y = math.radians(params.start_right.y) if params.start_right.y is not None else None
            poses[1, 2] = z
            poses[1, 3] = r if r is not None else poses[1, 3]
            poses[1, 4] = p if p is not None else poses[1, 4]
            poses[1, 5] = y if y is not None else poses[1, 5]

        if 'end_left' in params:
            # fix the left gripper end pose
            z = params.end_left.z
            r = math.radians(params.end_left.r) if params.end_left.r is not None else None
            p = math.radians(params.end_left.p) if params.end_left.p is not None else None
            y = math.radians(params.end_left.y) if params.end_left.y is not None else None
            poses[2, 2] = z
            poses[2, 3] = r if r is not None else poses[2, 3]
            poses[2, 4] = p if p is not None else poses[2, 4]
            poses[2, 5] = y if y is not None else poses[2, 5]

        if 'end_right' in params:
            # fix the right gripper end pose
            z = params.end_right.z
            r = math.radians(params.end_right.r) if params.end_right.r is not None else None
            p = math.radians(params.end_right.p) if params.end_right.p is not None else None
            y = math.radians(params.end_right.y) if params.end_right.y is not None else None
            poses[3, 2] = z
            poses[3, 3] = r if r is not None else poses[3, 3]
            poses[3, 4] = p if p is not None else poses[3, 4]
            poses[3, 5] = y if y is not None else poses[3, 5]

        return poses

    def _edge_pick_fixer(self,
                    pts_xyz_raw: np.ndarray,
                    poses: np.ndarray,
                    params: DictConfig) -> np.ndarray:
        """
        fix gripper translation based on contour direction (vertical to the contour)
        input params: poses: (4, 6) ndarray
        # each row in poses: (x, y, z) + (r, p, y)
        """

        # re-project the point cloud in virtual space into depth map in camera space, then transform it into mask map

        enable = params.enable
        if not enable:
            return poses
        fix_translation = params.fix_translation
        fix_rotation = params.fix_rotation
        virtual_offset = params.virtual_offset
        override_place = getattr(params, 'override_place', False)
        
        poses_bak = poses.copy()

        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd_trans_t = o3d.t.geometry.PointCloud(device)
        pcd_trans_t.point.positions = o3d.core.Tensor(pts_xyz_raw, dtype, device)
        pcd_trans_t.point.colors = o3d.core.Tensor(np.zeros_like(pts_xyz_raw), dtype, device)
        virtual_extrinsics = self.experiment.transforms.virtual_extrinsic_dummy
        rgbd_reproj = pcd_trans_t.project_to_rgbd_image(width=self.experiment.transforms.virtual_intrinsic.width,
                                                        height=self.experiment.transforms.virtual_intrinsic.height,
                                                        intrinsics=self.experiment.transforms.virtual_intrinsic.intrinsic_matrix,
                                                        extrinsics=virtual_extrinsics,
                                                        depth_scale=1.0,
                                                        depth_max=2.0)
        rgb, depth = np.asarray(rgbd_reproj.color.to_legacy()), np.asarray(rgbd_reproj.depth.to_legacy())[..., None]
        
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[depth > 0] = 255
        if self.args.debug:
            cv2.imshow("mask_raw", mask)
            cv2.waitKey()
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)
        if self.args.debug:
            cv2.imshow("mask_morphology", mask)
            cv2.waitKey()

        # find the biggest contour
        mask_image = mask.copy()
        contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_list = [cv2.contourArea(c) for c in contours]
        biggest_contour_index = np.argmax(area_list)
        
        result_mask = np.zeros_like(mask)
        cv2.fillPoly(result_mask, [contours[biggest_contour_index]], 255)
        biggest_contour = contours[biggest_contour_index][:, 0, :]
        
        # find uv coordinates in mask map for 3D grasp points
        left_grasp_point_uv = point_to_depth_map_location(poses[0, :3], pinhole_camera_intrinsic=self.experiment.transforms.virtual_intrinsic, extrinsics=virtual_extrinsics)
        right_grasp_point_uv = point_to_depth_map_location(poses[1, :3], pinhole_camera_intrinsic=self.experiment.transforms.virtual_intrinsic, extrinsics=virtual_extrinsics)
        if self.args.debug:
            result_mask_ = result_mask.copy()
            cv2.circle(result_mask_, (left_grasp_point_uv[0], left_grasp_point_uv[1]), 5, 100, -1)
            cv2.circle(result_mask_, (right_grasp_point_uv[0], right_grasp_point_uv[1]), 5, 100, -1)
            cv2.imshow("mask_final", result_mask_)
            cv2.waitKey()
        # find the point closet to the contour
        dist1 = np.linalg.norm(biggest_contour - left_grasp_point_uv, axis=1)
        idx1 = np.argmin(dist1)
        dist2 = np.linalg.norm(biggest_contour - right_grasp_point_uv, axis=1)
        idx2 = np.argmin(dist2)
        
        if fix_translation:
            grasp_point_uv = [left_grasp_point_uv, right_grasp_point_uv]
            closest_idx = [idx1, idx2]
            VIRTUAL_THRESHOLD = 0.2
            MAX_IDX_OFFSET = 5
            for i in range(2):
                if result_mask[grasp_point_uv[i][1], grasp_point_uv[i][0]] == 0:
                    # if the grasp point is outside the mask, find the nearest point inside the mask
                    logger.warning(f"Grasp point {i} is outside the contour, find the nearest point inside the contour")
                    # TODO: better implementation on z axis
                    # poses[i][:3] = depth_to_point(
                    #     (biggest_contour[closest_idx[i]][0], biggest_contour[closest_idx[i]][1]),
                    #     depth[biggest_contour[closest_idx[i]][1], biggest_contour[closest_idx[i]][0], 0],
                    #     self.experiment.transforms.virtual_intrinsic,
                    #     virtual_extrinsics
                    # )
                else:
                    nearner_contour_point = depth_to_point(
                        (biggest_contour[closest_idx[i]][0], biggest_contour[closest_idx[i]][1]),
                        depth[biggest_contour[closest_idx[i]][1], biggest_contour[closest_idx[i]][0], 0],
                        self.experiment.transforms.virtual_intrinsic,
                        virtual_extrinsics
                    )
                    dist = np.linalg.norm(poses[i][:2] - nearner_contour_point[:2])
                    if dist > VIRTUAL_THRESHOLD:
                        logger.warning(f"Grasp point {i} is inside the contour, and the distance {dist} is larger than threshold, skip auto-fixing")
                        continue

                for idx_offset in range(1, MAX_IDX_OFFSET):
                    try:
                        prev_p = biggest_contour[closest_idx[i] - idx_offset, :]
                        next_p = biggest_contour[(closest_idx[i] + idx_offset) % biggest_contour.shape[0], :]
                        prev_point = depth_to_point(
                            (prev_p[0], prev_p[1]),
                            depth[prev_p[1], prev_p[0], 0],
                            self.experiment.transforms.virtual_intrinsic,
                            virtual_extrinsics
                        )
                        next_point = depth_to_point(
                            (next_p[0], next_p[1]),
                            depth[next_p[1], next_p[0], 0],
                            self.experiment.transforms.virtual_intrinsic,
                            virtual_extrinsics
                        )

                        tangent_xy_dir = (next_point[:2] - prev_point[:2]) / \
                                         np.linalg.norm(next_point[:2] - prev_point[:2] + 1e-5)
                        perpendicular_xy_dir = np.stack([- tangent_xy_dir[1], tangent_xy_dir[0]])

                        for direction in [1, -1]:
                            new_pose = poses[i].copy()
                            new_pose[:2] += direction * virtual_offset * perpendicular_xy_dir
                            new_pose_uv = point_to_depth_map_location(new_pose[:3], pinhole_camera_intrinsic=self.experiment.transforms.virtual_intrinsic, extrinsics=virtual_extrinsics)
                            if result_mask[new_pose_uv[1], new_pose_uv[0]] == 255:
                                if self.args.debug:
                                    visualize_pc_and_grasp_points(pts_xyz_raw, left_pick_point=poses[i][:3], right_pick_point=new_pose[:3])
                                poses[i] = new_pose
                                logger.warning(f"Auto-fix gripper translation for grasp point {i} with direction {direction}")
                                break
                        break
                    except:
                        continue

        if fix_rotation:
            closest_idx = [idx1, idx2]
            ROT_OFFSET = 2
            for i in range(2):
                prev_p = biggest_contour[closest_idx[i] - ROT_OFFSET, :]
                next_p = biggest_contour[(closest_idx[i] + ROT_OFFSET) % biggest_contour.shape[0], :]

                prev_point = depth_to_point(
                                (prev_p[0], prev_p[1]),
                                depth[prev_p[1], prev_p[0], 0],
                                self.experiment.transforms.virtual_intrinsic,
                                virtual_extrinsics
                            )
                next_point = depth_to_point(
                    (next_p[0], next_p[1]),
                    depth[next_p[1], next_p[0], 0],
                    self.experiment.transforms.virtual_intrinsic,
                    virtual_extrinsics
                )

                tangent_xy_dir = (next_point[:2] - prev_point[:2]) / np.linalg.norm(next_point[:2] - prev_point[:2])
                theta = math.atan(tangent_xy_dir[1] / (tangent_xy_dir[0] + 1e-6))
                poses[i][4] = - theta
            
            poses[:2, 4] += math.pi
            if override_place:
                poses[2:, 4] = poses[:2, 4]
            
        # restore z-axis
        poses[:, 2] = poses_bak[:, 2]

        return poses

    def _search_pick_p_y_fixer(self,
                        pts_xyz_raw: np.ndarray,
                        poses: np.ndarray,
                        action_type: ActionTypeDef,
                        vertical_vector: Optional[np.ndarray],
                        delta_length: float,
                        params: DictConfig) -> np.ndarray:
        """
        search for the executable p angle for the gripper
        input params: poses: (4, 6) ndarray
        # each row in poses: (x, y, z) + (r, p, y)
        # poses definition:
        """
        enable = params.enable
        if not enable:
            return poses
        if params.p_choice_left is not None:
            p_choice_left = [math.radians(p) for p in params.p_choice_left]
        else:
            p_choice_left = [poses[0, 4]]
        if params.p_choice_right is not None:
            p_choice_right = [math.radians(p) for p in params.p_choice_right]
        else:
            p_choice_right = [poses[1, 4]]
        search_range = params.search_range
        search_range = [math.radians(search_range[0]), math.radians(search_range[1])]
        search_num = params.search_num
        search_order = params.search_order
        # search from mean value, and then expand the search range
        interval = (search_range[1] - search_range[0]) / (search_num - 1)
        mean_p = (search_range[0] + search_range[1]) / 2
        ps = [mean_p + math.pow(-1, i) * interval * ((i + 1) // 2) for i in range(search_num)]
        if search_order == 'side':
            search_idx_pair = reversed(sorted(list(itertools.product(range(search_num), repeat=2)), key=lambda x: min(x)))
        elif search_order == 'center':
            search_idx_pair = sorted(list(itertools.product(range(search_num), repeat=2)), key=lambda x: max(x))
        elif search_order == 'random':
            search_idx_pair = list(itertools.product(range(search_num), repeat=2))
            random.shuffle(search_idx_pair)
        elif search_order == 'grid':
            grid_y = params.grid.y
            increase_order = list(range(search_num-2, 0, -2)) + list(range(0, search_num, 2))
            normal_order = list(range(search_num))
            decrease_order = list(reversed(increase_order))
            if poses[0, 1] < grid_y[0]:
                order_left = decrease_order
            elif grid_y[0] < poses[0, 1] < grid_y[1]:
                order_left = normal_order
            else:
                order_left = increase_order
            if poses[1, 1] < grid_y[0]:
                ps_right = increase_order
            elif grid_y[0] < poses[1, 1] < grid_y[1]:
                ps_right = normal_order
            else:
                ps_right = decrease_order
            search_idx_pair = list(itertools.product(order_left, ps_right))
        else:
            raise ValueError(f"search_order {search_order} is not supported!")
        if self.args.use_parallel_checker:
            current_joint_space_state = self.experiment.controller.actuator.getJointSpaceState()
            timestamp = time.time()
            tasks = [deepcopy((os.getpid(), timestamp, poses, p_choice_left, p_choice_right, ps, idx_pair, action_type, vertical_vector, delta_length, current_joint_space_state)) for idx_pair in search_idx_pair]
            for task in tasks:
                self.checker_task_queue.put(task)
            logger.debug(f'Finish creating tasks for parallel checker!')

            results = []
            while True:
                result = self.checker_result_queue.get()
                if result[0] == timestamp:
                    if result[2] is None:
                        logger.info(f"Found executable p angle for gripper: {result[1][0, 5]}, {result[1][1, 5]}")
                        return result[1]
                    else:
                        results.append(result)
                        if len(results) == len(tasks):
                            break
        else:
            for p_left, p_right in itertools.product(p_choice_left, p_choice_right):
                poses[0, 4] = p_left
                poses[1, 4] = p_right
                
                for idx_pair in search_idx_pair:
                    for i in range(2):
                        poses[i, 5] = ps[idx_pair[i]]
                        if vertical_vector is not None:
                            if poses[i, 4] > np.pi:
                                sign = -1
                            else:
                                sign = 1
                            if np.abs(vertical_vector[0]) > np.abs(vertical_vector[1]):
                                if vertical_vector[0] > 0:
                                    sign = -sign
                                else:
                                    sign = sign
                            else:
                                if vertical_vector[1] > 0:
                                    sign = sign
                                else:
                                    sign = -sign
                                pass
                            logger.debug(
                                f"p_left: {p_left}, p_right: {p_right}, sign: {sign}, vertical_vector: {vertical_vector}, delta_length: {delta_length}, sin: {np.sin(poses[i, 5])}, sweep fixing offset： {sign * vertical_vector * delta_length * np.sin(poses[i, 5])}")
                            poses[i, :2] += sign * vertical_vector * delta_length * np.sin(poses[i, 5])

                    euler_type = 'XZY' if action_type == ActionTypeDef.SWEEP else 'XZX'
                    pick_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses[0], euler_type=euler_type)
                    pick_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses[1], euler_type=euler_type)
                    place_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses[2], euler_type=euler_type)
                    place_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses[3], euler_type=euler_type)

                    _, err = self.experiment.is_action_executable(action_type, (pick_1, pick_2, place_1, place_2))
                    if err is None:
                        logger.info(f"Found executable p angle for gripper: {poses[0, 5]}, {poses[1, 5]}")
                        return poses
                    else:
                        logger.warning(f"p angle {poses[0, 5]}, {poses[1, 5]} is not executable, error: {err}")

        logger.warning(f"Cannot find executable p angle for gripper, use the original poses")
        return poses

    @staticmethod
    def _search_pick_p_y_fixer_worker(option: DictConfig, task_queue: Queue, result_queue: Queue):
        parent_pid = None
        option.compat.camera.use_file_camera = True
        option.compat.grasp_checker.enable = False
        experiment = ExperimentReal(option)

        while True:
            while True:
                try:
                    task = task_queue.get(timeout=5)
                    break
                except:
                    if parent_pid is not None and not psutil.pid_exists(parent_pid):
                        exit(-1)
                    else:
                        continue
            if task is None:
                break

            parent_pid, timestamp, poses, p_choice_left, p_choice_right, ps, idx_pair, action_type, vertical_vector, delta_length, current_joint_space_state = task
            left_joints_value, right_joints_value = np.split(np.array(current_joint_space_state), 2)
            
            for p_left, p_right in itertools.product(p_choice_left, p_choice_right):
                setJointSpaceState(experiment.controller.actuator, left_joints_value, right_joints_value, delay=0.0)

                poses[0, 4] = p_left
                poses[1, 4] = p_right
                
                for idx in range(2):
                    poses[idx, 5] = ps[idx_pair[idx]]
                    if vertical_vector is not None:
                        if poses[i, 4] > np.pi:
                            sign = -1
                        else:
                            sign = 1
                        if np.abs(vertical_vector[0]) > np.abs(vertical_vector[1]):
                            if vertical_vector[0] > 0:
                                sign = -sign
                            else:
                                sign = sign
                        else:
                            if vertical_vector[1] > 0:
                                sign = sign
                            else:
                                sign = -sign
                            pass
                        logger.debug(
                            f"p_left: {p_left}, p_right: {p_right}, sign: {sign}, vertical_vector: {vertical_vector}, delta_length: {delta_length}, sin: {np.sin(poses[i, 5])}, sweep fixing offset： {sign * vertical_vector * delta_length * np.sin(poses[i, 5])}")
                        poses[i, :2] += sign * vertical_vector * delta_length * np.sin(poses[i, 5])

                euler_type = 'XZY' if action_type == ActionTypeDef.SWEEP else 'XZX'
                pick_1 = experiment.transforms.virtual_pose_to_world_pose(poses[0], euler_type=euler_type)
                pick_2 = experiment.transforms.virtual_pose_to_world_pose(poses[1], euler_type=euler_type)
                place_1 = experiment.transforms.virtual_pose_to_world_pose(poses[2], euler_type=euler_type)
                place_2 = experiment.transforms.virtual_pose_to_world_pose(poses[3], euler_type=euler_type)

                _, err = experiment.is_action_executable(action_type, (pick_1, pick_2, place_1, place_2))
                if err is None:
                    break

            logger.debug(f"searching p angle for gripper, p1: {poses[0, 5]}, p2: {poses[1, 5]}, error: {err}")
            result_queue.put((timestamp, poses, err))

    def choose_random_point_for_lift(self, virtual_pc_xyz: np.ndarray) -> Tuple[ActionMessage, Optional[ExceptionMessage]]:
        action_type = ActionTypeDef.LIFT
        if not self.experiment.option.compat.use_real_robots:
            logger.exception('Lift action is only supported in real robots')

        fixer_params = self.args.action_fixer.get(ActionTypeDef.to_string(action_type), None)
        assert fixer_params is not None, f"Action type {action_type} is not supported in action fixer!"
        assert fixer_params.other_params.max_trial_num > 0, "max_trial_num should be larger than 0"


        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd_trans_t = o3d.t.geometry.PointCloud(device)
        pcd_trans_t.point.positions = o3d.core.Tensor(virtual_pc_xyz, dtype, device)
        pcd_trans_t.point.colors = o3d.core.Tensor(np.zeros_like(virtual_pc_xyz), dtype, device)
        virtual_extrinsics = self.experiment.transforms.virtual_extrinsic_dummy
        rgbd_reproj = pcd_trans_t.project_to_rgbd_image(width=self.experiment.transforms.virtual_intrinsic.width,
                                                        height=self.experiment.transforms.virtual_intrinsic.height,
                                                        intrinsics=self.experiment.transforms.virtual_intrinsic.intrinsic_matrix,
                                                        extrinsics=virtual_extrinsics,
                                                        depth_scale=1.0,
                                                        depth_max=2.0)
        rgb, depth = np.asarray(rgbd_reproj.color.to_legacy()), np.asarray(rgbd_reproj.depth.to_legacy())[..., None]

        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[depth > 0] = 255
        if self.args.debug:
            cv2.imshow("mask_raw", mask)
            cv2.waitKey(0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)
        if self.args.debug:
            cv2.imshow("mask_morphology", mask)
            cv2.waitKey(0)

        # find the biggest contour
        mask_image = mask.copy()
        contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_list = [cv2.contourArea(c) for c in contours]
        biggest_contour_index = np.argmax(area_list)

        result_mask = np.zeros_like(mask)
        cv2.fillPoly(result_mask, [contours[biggest_contour_index]], 255)
        biggest_contour = contours[biggest_contour_index][:, 0, :]

        for trial_idx in range(fixer_params.other_params.max_trial_num):
            # randomly select a point on the contour
            random_idx = np.random.choice(biggest_contour.shape[0], 1)

            virtual_chosen_pt = depth_to_point(
                (biggest_contour[random_idx, 0], biggest_contour[random_idx, 1]),
                depth[biggest_contour[random_idx, 1], biggest_contour[random_idx, 0], 0],
                self.experiment.transforms.virtual_intrinsic,
                virtual_extrinsics
            )

            virtual_pose6d = np.concatenate([virtual_chosen_pt, np.zeros_like(virtual_chosen_pt)], axis=-1)  # (x, y, z, r, p, y)
            # repeat the pose for 4 times
            poses6d = np.tile(virtual_pose6d, (4, 1))  # (4, 6)
            # fix the pose based on predefined parameters
            poses6d = self._predefined_fixer(virtual_pc_xyz, poses6d, params=fixer_params.predefined_fixer_params)
            poses6d = self._edge_pick_fixer(virtual_pc_xyz, poses6d, params=fixer_params.edge_pick_fixer_params)
            poses6d = self._search_pick_p_y_fixer(virtual_pc_xyz, poses6d, action_type, vertical_vector=None, delta_length=0.0,
                                                params=fixer_params.search_pick_p_y_fixer_params)
            # check whether the action is executable
            pick_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[0])
            pick_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[1])
            place_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[2])
            place_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[3])
            # check whether the action is executable
            transforms, err = self.experiment.is_action_executable(action_type, (pick_1, pick_2, place_1, place_2))
            if err is None:
                if 'pick_left' in transforms:
                    return ActionMessage(action_type=action_type,
                                         pick_points=[transforms['pick_left'], None],
                                         joints_value_list=[transforms['joints_value_left_list'], None]), None
                elif 'pick_right' in transforms:
                    return ActionMessage(action_type=action_type,
                                         pick_points=[None, transforms['pick_right']],
                                         joints_value_list=[None, transforms['joints_value_right_list']]), None
            else:
                logger.warning(f'Can not find executable lift action for {transforms}')
            
        return (ActionMessage(action_type=action_type), err)
                
    def choose_points_for_drag(self, virtual_pc_xyz: np.ndarray,
                               mode: str = 'crumpled',
                               last_action: ActionMessage = None) -> Tuple[ActionMessage, Optional[ExceptionMessage]]:
        """
        Get the best pick and place poses for drag action with two robot arms in virtual space.
        Select pick points from the line between the robot base and the target pick points (too far to reach)
        
        :param virtual_pc_xyz: point cloud in virtual space
        :param mode: crumpled or folded_once
        :param last_action: last action message

        return ActionMessage, ExceptionMessage
        return None, ExceptionMessage if action is not executable
        """
        assert mode in ('crumpled', 'folded_once'), 'mode should be crumpled or folded_once'
        action_type = ActionTypeDef.DRAG
        if not self.experiment.option.compat.use_real_robots:
            logger.exception('Drag action is only supported in real robots')
        
        drag_pred_rule = getattr(self.args, 'drag_pred_rule', None)
        if drag_pred_rule is None:
            logger.exception('drag_pred_rule is not defined in the config file!')

        if mode == 'crumpled':
            drag_rule_param = drag_pred_rule.drag_rule_param_in_crumpled_mode
        elif mode == 'folded_once':
            drag_rule_param = drag_pred_rule.drag_rule_param_in_folded_once_mode
        else:
            logger.exception('mode should be crumpled or folded_once')
            raise NotImplementedError
        
        fixer_params = self.args.action_fixer.get(ActionTypeDef.to_string(action_type), None)
        assert fixer_params is not None, f"Action type {action_type} is not supported in action fixer!"
        assert fixer_params.other_params.max_trial_num > 0, "max_trial_num should be larger than 0"

        use_ref_target_points = False
        if last_action is not None and last_action.action_type == ActionTypeDef.FLING:
            # use the pick points of the last action as target points
            left_pick_in_world, right_pick_in_world = last_action.pick_points[0], last_action.pick_points[1]
            # transform pick points from world space to virtual space
            left_target_point = self.experiment.transforms.world_pose_to_virtual_pose(left_pick_in_world).translation
            right_target_point = self.experiment.transforms.world_pose_to_virtual_pose(right_pick_in_world).translation
            use_ref_target_points = True
            logger.debug(f"Use the pick points of the last action as target points: {left_target_point}, {right_target_point}")
        else:
            centroid = np.mean(virtual_pc_xyz, axis=0)
            left_target_point = centroid.copy()
            right_target_point = centroid.copy()
            use_ref_target_points = False
            logger.debug(f"Use the centroid as target points: {left_target_point}, {right_target_point}")
        pick_points_ratio = drag_rule_param.pick_points_ratio

        drag_towards_robot = None
        if left_target_point[1] >= drag_rule_param.y_threshold or right_target_point[1] >= drag_rule_param.y_threshold:
            left_reference_point_xy = np.array(drag_rule_param.reference_points_xy.left_close)
            right_reference_point_xy = np.array(drag_rule_param.reference_points_xy.right_close)
            drag_towards_robot = True
        else:
            left_reference_point_xy = np.array(drag_rule_param.reference_points_xy.left_far)
            right_reference_point_xy = np.array(drag_rule_param.reference_points_xy.right_far)
            drag_towards_robot = False

        left_dist_all = np.linalg.norm(virtual_pc_xyz[:, :2] - left_reference_point_xy[None, :], axis=1)
        right_dist_all = np.linalg.norm(virtual_pc_xyz[:, :2] - right_reference_point_xy[None, :], axis=1)
        left_reference_point = virtual_pc_xyz[np.argmin(left_dist_all), :]
        right_reference_point = virtual_pc_xyz[np.argmin(right_dist_all), :]
        for trial_idx in range(fixer_params.other_params.max_trial_num):            
            if use_ref_target_points:
                # update the reference points based on the pick points of the last action
                left_grasp_reference_point = left_reference_point + (left_target_point - left_reference_point) * pick_points_ratio
                right_grasp_reference_point = right_reference_point + (right_target_point - right_reference_point) * pick_points_ratio
            else:
                # update the reference points based on the centroid
                left_grasp_reference_point = left_reference_point.copy()
                right_grasp_reference_point = right_reference_point.copy()
                left_grasp_reference_point[1] = left_reference_point[1] + (left_target_point[1] - left_reference_point[1]) * pick_points_ratio
                right_grasp_reference_point[1] = right_reference_point[1] + (right_target_point[1] - right_reference_point[1]) * pick_points_ratio
            left_dist_all = np.linalg.norm(virtual_pc_xyz[:, :2] - left_grasp_reference_point[:2], axis=1)
            right_dist_all = np.linalg.norm(virtual_pc_xyz[:, :2] - right_grasp_reference_point[:2], axis=1)
            left_pick_point = virtual_pc_xyz[np.argmin(left_dist_all), :]
            right_pick_point = virtual_pc_xyz[np.argmin(right_dist_all), :]

            left_place_point = left_pick_point.copy()
            right_place_point = right_pick_point.copy()

            if drag_towards_robot:
                left_place_point[1] = drag_rule_param.place_points_y_close
                right_place_point[1] = drag_rule_param.place_points_y_close
            else:
                left_place_point[1] = drag_rule_param.place_points_y_far
                right_place_point[1] = drag_rule_param.place_points_y_far

            if drag_rule_param.place_to_center:
                place_center = (left_place_point + right_place_point) / 2
                place_center[0] = 0.  # maker sure that the x-axis coordinate is at center

            pose = np.stack([left_pick_point, right_pick_point, left_place_point, right_place_point], axis=0)
            poses6d = np.concatenate([pose, np.zeros_like(pose)], axis=-1)  # (4, 6)
            
            poses6d = self._predefined_fixer(virtual_pc_xyz, poses6d, params=fixer_params.predefined_fixer_params)
            poses6d = self._edge_pick_fixer(virtual_pc_xyz, poses6d, params=fixer_params.edge_pick_fixer_params)
            poses6d = self._search_pick_p_y_fixer(virtual_pc_xyz, poses6d, action_type, vertical_vector=None, delta_length=0.0,
                                                params=fixer_params.search_pick_p_y_fixer_params)
            # check whether the action is executable
            pick_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[0])
            pick_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[1])
            place_1 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[2])
            place_2 = self.experiment.transforms.virtual_pose_to_world_pose(poses6d[3])
            # check whether the action is executable
            transforms, err = self.experiment.is_action_executable(action_type, (pick_1, pick_2, place_1, place_2))
        
            if err is None:
                return ActionMessage(action_type=action_type,
                                    pick_points=[transforms['pick_left'], transforms['pick_right']],
                                    place_points=[transforms['place_left'], transforms['place_right']]), None
            else:
                logger.warning(f"Drag action trial {trial_idx}! pick_points_ratio: {pick_points_ratio}")
            pick_points_ratio *= drag_rule_param.pick_points_ratio_decay
        
        return (ActionMessage(action_type=action_type), err)


    def predict_object_state(self,
                                obs_msg: ObservationMessage,
                                running_seed: int = None, 
                                ignore_manual_operation: bool = False) -> ObjectState:
        "Predict raw action type by the action classifier, this action type could be changed later if required"
        # TODO: change this to state prediction for condition update in state machine
        if not ignore_manual_operation and self.args.manual_operation.enable:
            object_state = self.get_object_state_manually(obs_msg)
        else:
            raise NotImplementedError
        
        return object_state
    
    def get_object_state_manually(self, obs_msg: ObservationMessage) -> ObjectState:
        pcd_xyz_unmasked = obs_msg.raw_virtual_pcd
        remote_args = self.args.manual_operation.remote_args
        if remote_args.enable:
            raise NotImplementedError
        else:
            __DISPLAY_OPTIONS__ = [
                "down",
                "up",
                "left",
                "right",
                "unknown",
            ]
            __DISPLAY_OPTIONS_MAPPING__ = [
                GarmentSmoothingStyle.DOWN,
                GarmentSmoothingStyle.UP,
                GarmentSmoothingStyle.LEFT,
                GarmentSmoothingStyle.RIGHT,
                GarmentSmoothingStyle.UNKNOWN
            ]

            Console().print(
                        "[instruction] observe the object state, then press Esc to close the window")

            o3d.visualization.draw_geometries([pcd_xyz_unmasked])
            object_state = ObjectState()
            
            organized = py_cli_interaction.must_parse_cli_bool("organized enough?", default_value=False)
            if organized:
                object_state.general_object_state = GeneralObjectState.ORGANIZED

                base = 1
                smoothing_style_idx = py_cli_interaction.must_parse_cli_sel("smoothing style?",
                                                                            __DISPLAY_OPTIONS__, min=base) - base
                object_state.garment_smoothing_style = __DISPLAY_OPTIONS_MAPPING__[smoothing_style_idx]
            else:
                object_state.general_object_state = GeneralObjectState.DISORDERED
                object_state.garment_smoothing_style = GarmentSmoothingStyle.UNKNOWN
            
        return object_state

    def predict_action(self, obs_message: ObservationMessage, action_type: ActionTypeDef = None,
                       vis: bool = False, running_seed: int = None,
                       ) -> Tuple[PredictionMessage, ActionMessage, Optional[ExceptionMessage]]:
        pts_xyz_raw, pts_xyz_unmasked = obs_message.valid_virtual_pts, obs_message.raw_virtual_pts
        pcd_xyz_raw, pcd_xyz_unmasked = obs_message.valid_virtual_pcd, obs_message.raw_virtual_pcd
        timing = {'start_timestamp': time.time()}

        # TODO: better implementation
        pts_xyz_raw = self.remove_ood_points(pts_xyz_raw)
        pts_xyz_centered = self.conduct_zero_center_and_scale(pts_xyz_raw)
        while True:
            try:
                pts_xyz_batch, coords, feat = self.transform_input(pts_xyz_centered, seed=running_seed)
                break
            except Exception as e:
                logger.error(f'Error in transforming input: {e}. Retry.')
        pts_xyz_numpy = pts_xyz_batch[0].cpu().numpy()
        timing['pre_processing_timestamp'] = time.time()
        timing['pre_processing'] = timing['pre_processing_timestamp'] - timing['start_timestamp']
        if action_type in (ActionTypeDef.FOLD_1_1, ActionTypeDef.FOLD_1_2, ActionTypeDef.FOLD_2):
            raise NotImplementedError
        else:
            assert self.model_version >= 'v6' or 'diffusion' in self.model_version, 'mask is only supported in model version >= v6'
            prediction_message: PredictionMessage = self.model.predict(pts_xyz_batch, coords, feat,
                                                                       action_type=action_type,
                                                                       return_timing=True)
            prediction_message.grasp_point_all = self.undo_zero_center_and_scale(prediction_message.grasp_point_all)
            pts_xyz_numpy = self.undo_zero_center_and_scale(pts_xyz_numpy)
            prediction_message.pc_xyz = pts_xyz_numpy
            timing['model_prediction_timestamp'] = time.time()
            timing['nn_prediction'] = timing['model_prediction_timestamp'] - timing['pre_processing_timestamp']

        action_type = prediction_message.action_type if action_type is None else action_type
        logger.info(f"action_type={ActionTypeDef.to_string(action_type)}")
        # enumerate possible actions and check whether the action is executable
        verbose = True
        enable_drag_for_fold2 = False
        err = None
        if action_type == ActionTypeDef.DONE:
            return prediction_message, ActionMessage(action_type=action_type, 
                                                     extra_params={'score': None, 'timing': timing, 'idxs': [-1, -1]}), None
        else:
            if action_type == ActionTypeDef.FLING and self.args.fling_override:
                # manually select the best grasp-points by user
                # use unmasked colored point cloud for fling action override
                pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy, pcd_xyz_unmasked=pcd_xyz_unmasked)
                idxs = [-1, -1]
            elif action_type == ActionTypeDef.SWEEP and getattr(self.args, 'sweep_override', False) or \
                    action_type == ActionTypeDef.SINGLE_PICK_AND_PLACE and getattr(self.args, 'single_pick_and_place_override', False):
                # manually select the best grasp-points by user
                # use unmasked colored point cloud for sweep or single_pick_and_place_override action override
                while True:
                    pcd_xyz_unmasked = self.experiment.mark_pcd_with_workspace(pcd_xyz_unmasked)
                    pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy, pcd_xyz_unmasked=pcd_xyz_unmasked)

                    if err is None:
                        break
                    else:
                        Console().print(
                            "[instruction] poses are unreachable, please re-select")

                idxs = [-1, -1]
            elif action_type == ActionTypeDef.STRAIGHTEN_AND_PLACE and getattr(self.args, 'straighten_and_place_override', False):
                # manually select the best grasp-points by user
                # use unmasked colored point cloud for straighten_and_place_override action override
                while True:
                    pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy, pcd_xyz_unmasked=pcd_xyz_unmasked)

                    if err is None:
                        break
                    else:
                        Console().print(
                            "[instruction] poses are unreachable, please re-select")

                idxs = [-1, -1]
            elif action_type == ActionTypeDef.PICK_AND_PLACE and self.args.pick_and_place_override:
                pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy, pts_xyz_unmasked=pts_xyz_unmasked)
                idxs = [-1, -1]
            elif action_type == ActionTypeDef.FOLD_1_1 and self.args.fold1_override:
                pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy)
                idxs = [-1, -1]
            elif action_type == ActionTypeDef.FOLD_2 and self.args.fold2_override:
                pc_xyz_world, poses, transforms, err = self.get_policy_from_picker(action_type, pts_xyz_raw, pts_xyz_numpy, pts_xyz_unmasked=pts_xyz_unmasked)
                idxs = [-1, -1]
            else:
                timing['before_action_filter_timestamp'] = time.time()
                # use model prediction
                pc_xyz_world, poses, transforms, idxs, action_type, err = \
                    self.filter_model_prediction(prediction_message.action_iterator,
                                                action_type,
                                                pts_xyz_raw,
                                                pts_xyz_numpy,
                                                prediction_message.grasp_point_all)
                timing['action_filter_timestamp'] = time.time()
                timing['action_filter'] = timing['action_filter_timestamp'] - timing['before_action_filter_timestamp']

        timing['total_time'] = time.time() - timing['start_timestamp']

        if transforms is None:
            if verbose:
                logger.warning(f'Could not find a valid pose (end of iterator).')
            if self.args.drag_for_fold1 and action_type == ActionTypeDef.FOLD_1_1:
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FOLD_1_1,
                                                         extra_params={'score': None, 'timing': timing,
                                                                       'idxs': [-1, -1]}), \
                    ExceptionMessage("The best pose is not valid.")

            elif self.args.drag_for_fold2 and action_type == ActionTypeDef.FOLD_2:
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FOLD_2,
                                                         extra_params={'score': None, 'timing': timing,
                                                                       'idxs': [-1, -1]}), \
                    ExceptionMessage("The best pose is not valid.")
            else:
                # TODO: handle fail action
                logger.error(f'Could not find valid pose for {action_type}!')
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FAIL,
                                                         extra_params={'score': None, 'timing': timing, 'idxs': [-1, -1]}), \
                    ExceptionMessage("Could not find a valid pose (end of iterator).")
        else:
            logger.info(f'Predict action type : {action_type}!')
            if vis and transforms is not None:
                geometry_list = self.create_vis_geometries(transforms, pc_xyz_world)
                o3d.visualization.draw_geometries(geometry_list,
                                                  lookat=np.array([[0.5, 0., 0.]]).T,
                                                  up=np.array([[1., 0., 0.]]).T,
                                                  front=np.array([[0., 0., 1.]]).T, zoom=1.0)

            return (
                prediction_message, 
                ActionMessage(
                    action_type=action_type,
                    object_type=self.experiment.option.compat.object_type,
                    pick_points=[transforms['pick_left'] if 'pick_left' in transforms else None,
                                 transforms['pick_right'] if 'pick_right' in transforms else None],
                    place_points=[transforms['place_left'] if 'place_left' in transforms else None,
                                  transforms['place_right'] if 'place_right' in transforms else None],
                    joints_value_list=[
                        transforms['joints_value_left_list'] if 'joints_value_left_list' in transforms else None,
                        transforms['joints_value_right_list'] if 'joints_value_right_list' in transforms else None
                    ],
                    extra_params={'score': None, 'timing': timing, 'idxs': idxs},
                    extra_action_params=transforms['extra_params'] if 'extra_params' in transforms else {},
                ),
                err
            )


    def get_policy_from_picker(self,
                               action_type: ActionTypeDef,
                               pts_xyz: np.ndarray,
                               pts_xyz_numpy: np.ndarray,
                               pts_xyz_unmasked: np.ndarray = None,
                               pcd_xyz: np.ndarray = None,
                               pcd_xyz_unmasked: np.ndarray = None,
                            ) -> (
            Tuple)[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[ExceptionMessage]]:
        remote_args = self.args.manual_operation.remote_args
        if remote_args.enable:
            assert self.args.manual_operation.enable, 'remote operation requires manual operation'
            pick_function = partial(
                remote_pick_n_points_from_pcd,
                host=remote_args.host,
                port=remote_args.anno_port,
                debug=remote_args.debug
            )
        else:
            Console().print(
                        "[instruction] Select ideal poses in order")
            pick_function = pick_n_points_from_pcd
        
        if pcd_xyz_unmasked is not None:
            virtual_pcd = pcd_xyz_unmasked
        elif pcd_xyz is not None:
            virtual_pcd = pcd_xyz
        elif pts_xyz_unmasked is not None:
            # use unmasked point cloud for pick-and-place action annotation
            virtual_pcd = o3d.geometry.PointCloud()
            virtual_pcd.points = o3d.utility.Vector3dVector(pts_xyz_unmasked)
        else:
            virtual_pcd = o3d.geometry.PointCloud()
            virtual_pcd.points = o3d.utility.Vector3dVector(pts_xyz)

        pts: List[np.ndarray] = None
        if action_type in [ActionTypeDef.FLING]:
            while True:
                pts, _, err = pick_function(virtual_pcd, 2)
                if err is None:
                    break
            poses = np.array([[*pts[0], 0.], [*pts[1], 0.], [0.,0.,0.,0.,], [0.,0.,0.,0.,]])
        elif action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
            while True:
                pts, _, err = pick_function(virtual_pcd, 2)
                if err is None:
                    break
            poses = np.array([[*pts[0], 0.], [*pts[0], 0.], [*pts[1], 0.], [*pts[1], 0.]])  # left and right action points are the same
        elif action_type in [ActionTypeDef.PICK_AND_PLACE, ActionTypeDef.FOLD_1_1, ActionTypeDef.FOLD_2, ActionTypeDef.STRAIGHTEN_AND_PLACE]:
            while True:
                pts, _, err = pick_function(virtual_pcd, 4)
                if err is None:
                    break
            poses = np.array([[*pts[0], 0.], [*pts[1], 0.], [*pts[2], 0.],[*pts[3], 0.]])
        else:
            return None, None, None, ExceptionMessage(NotImplementedError)

        pc_xyz_world = transform_point_cloud(pts_xyz_numpy, self.experiment.transforms.virtual_to_world_transform)
        poses_world = self.transform_output(poses, pts_xyz=pts_xyz_numpy, pts_xyz_raw=pts_xyz, action_type=action_type)
        transforms, err = self.experiment.is_action_executable(action_type, poses_world)

        return pc_xyz_world, poses, transforms, err


    def filter_model_prediction(self,
                                action_iterator,
                                action_type: ActionTypeDef,
                                pts_xyz_raw: np.ndarray,
                                pts_xyz_sampled: np.ndarray,
                                grasp_points_all: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, ActionTypeDef]:
        """
        filter all model predictions and find feasible actions
        """
        best_poses = None
        best_transforms = None
        best_idxs = None
        first_transforms = None
        has_found_best_pick_pts = False
        if self.args.vis_all_fling_pred and action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
            points_pair_num = grasp_points_all.shape[0] // 2
            labels = [str(i) for i in range(points_pair_num) for _ in range(2)]
            if action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                visualize_pc_and_grasp_points(pts_xyz_sampled, grasp_candidates=grasp_points_all, candidates_group_size=2, labels=labels, ordering=True)
            else:
                visualize_pc_and_grasp_points(pts_xyz_sampled, grasp_candidates=grasp_points_all, candidates_group_size=2, labels=labels)
        # transform point cloud from virtual space into world space
        pts_xyz_raw_world = transform_point_cloud(pts_xyz_raw, self.experiment.transforms.virtual_to_world_transform)
        # calculate probability for random exploration
        if self.experiment.option.strategy.random_exploration.enable:
            use_random_pred = np.random.random() < \
                                 self.experiment.option.strategy.random_exploration.random_explore_prob
            random_top_ratio = self.experiment.option.strategy.random_exploration.random_explore_top_ratio
            if use_random_pred:
                logger.warning(f'Using random exploration now! Trying to randomly choose from top {random_top_ratio * 100} '
                            f'% action poses...')
        else:
            use_random_pred = False
            random_top_ratio = 0.
        if self.model.random_select_diffusion_action_pair_for_inference:
            use_random_pred = True
            random_top_ratio = 1.0 / (self.model.state_head.num_pred_candidates * 2)
        if self.model.use_dpo_reward_for_inference or self.model.use_reward_prediction_for_inference:
            top_ratio = 1.0 / (self.model.state_head.num_pred_candidates * 2)
        else:
            top_ratio = 1.0
        # iterate all possible actions (sorted by scores, large-f
        # irst)
        for i, action_iterator_msg in enumerate(action_iterator(use_random_pred, random_top_ratio, top_ratio)):
            if self.args.vis_pred_order and i >= self.args.vis_pred_order_num:
                break
            poses, poses_nocs, idxs = (action_iterator_msg.poses_4d, action_iterator_msg.poses_nocs,
                                       action_iterator_msg.grasp_idxs)
            if action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                # TODO: better implementation
                poses[:, :3] = self.undo_zero_center_and_scale(poses[:, :3])
            idx1, idx2 = idxs  # left pick-point index, right pick-point index            
            nearest_pick_override = getattr(self.args, 'nearest_pick_override', False)
            poses_world = self.transform_output(poses,
                                                pts_xyz=pts_xyz_sampled,
                                                action_type=action_type,
                                                pts_xyz_raw=pts_xyz_raw,
                                                nearest_pick_override=nearest_pick_override)
            if self.args.vis_pred_order and action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                open3d_pose_dict = dict(lookat=np.array([[0.5, 0., 0.]]).T, up=np.array([[1., 0., 0.]]).T,
                                        front=np.array([[0., 0., 1.]]).T, zoom=1.0)
                visualize_pc_and_grasp_points(pts_xyz_raw_world,
                                              left_pick_point=poses_world[0].translation,
                                              right_pick_point=poses_world[1].translation,
                                              visualization_pose_dict=open3d_pose_dict)
            if i == 0:
                # judge whether the predicted action is executable,
                # and transforms it into world-space poses (represented by RigidTransform class)
                transforms, err = self.experiment.is_action_executable(action_type, poses_world, 
                                                                    return_detailed_err=
                                                                    self.args.drag_for_best_fling_pick_pts or self.args.drag_for_fold1 or self.args.drag_for_fold2)
                if action_iterator_msg.extra_params is not None:
                    transforms['extra_params'] = action_iterator_msg.extra_params
                # judge whether pick poses is reachable
                pick1, pick2, place1, place2 = poses_world
                
                if first_transforms is None:
                    first_transforms = transforms.copy()

                if err is None and best_transforms is None:
                    has_found_best_pick_pts = True
                    best_idxs = idxs
                    best_poses = poses.copy()
                    best_transforms = transforms.copy()
                    if not self.args.vis_pred_order:
                        break
                elif err is not None and not has_found_best_pick_pts:
                    if action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE] and self.args.drag_for_best_fling_pick_pts:
                        # we only check the best (first) candidates here if we want to trigger dragging
                        logger.warning(f'The best prediction of action {action_type} is not executable, '
                                    f'give up this action now....')
                        if not self.args.vis_pred_order:
                            break

        if best_transforms is None:
            if action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                best_transforms = first_transforms
            if action_type != ActionTypeDef.DONE and self.args.vis_err_actin:
                logger.debug(f'predicted poses: {transforms}')
                logger.error('Failed to find a valid action! Show visualization for debugging...')
                # visualize action poses even if they are un-executable
                geometry_list = self.create_vis_geometries(transforms, pts_xyz_raw_world)
                o3d.visualization.draw_geometries(geometry_list,
                                                lookat=np.array([[0.5, 0., 0.]]).T,
                                                up=np.array([[1., 0., 0.]]).T,
                                                front=np.array([[0., 0., 1.]]).T, zoom=1.0)
            else:
                if self.args.debug:
                    logger.debug(f'predicted poses: {best_transforms}')
        return pts_xyz_raw_world, best_poses, best_transforms, best_idxs, action_type, err

    def create_vis_geometries(self, transforms: dict, pc_xyz_world: np.ndarray, pc_offset: tuple = (0., 0., 0.)):
        input_pcd = o3d.geometry.PointCloud()
        input_pcd.points = o3d.utility.Vector3dVector(pc_xyz_world)
        input_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pc_xyz_world) * 0.5)

        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        left_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        left_robot.transform(self.experiment.transforms.left_robot_to_world_transform)
        right_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        right_robot.transform(self.experiment.transforms.right_robot_to_world_transform)

        grasp_point_list = []
        dir_point_list = []
        for key, transform in transforms.items():
            if isinstance(transform, RigidTransform):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025).translate(transform.translation)
            else:
                continue
            if key == 'pick_left':
                sphere.paint_uniform_color([0.9, 0.0, 0.0])  # dark red
            elif key == 'pick_right':
                sphere.paint_uniform_color([0., 0.0, 0.9])  # dark blue
            elif key == 'place_left':
                sphere.paint_uniform_color([0.5, 0.2, 0.2])  # light red
            elif key == 'place_right':
                sphere.paint_uniform_color([0.2, 0.2, 0.5])  # light blue
            grasp_point_list.append(sphere)

            # theta = transform.euler_angles[-1]
            # grasp_point_dir = np.array([math.cos(-theta), math.sin(-theta), 0])
            # start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015). \
            #     translate(transform.translation - grasp_point_dir * 0.03)
            # start_sphere.paint_uniform_color([0., 1., 0.])  # green
            # end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015). \
            #     translate(transform.translation + grasp_point_dir * 0.03)
            # end_sphere.paint_uniform_color([0., 0., 0.])  # black
            # dir_point_list.extend([start_sphere, end_sphere])

        geometry_list = [world, left_robot, right_robot, input_pcd] + grasp_point_list + dir_point_list
        # add offset for all geometries (only for visualization)
        geometry_list = [geometry.translate(pc_offset) for geometry in geometry_list]
        return geometry_list

    def init_test_info(self):
        self.test_info['current_steps'] = 0
        
    def update_test_info(self, status: str):
        # status: success, failed, action
        logger.info("-"*40)
        self.test_info['current_steps'] += 1
        if status == 'success' or status == 'failed':
            self.test_info[status] += 1
            if status == 'success':
                self.test_info['history_steps'].append(self.test_info['current_steps'] - 1)
            logger.info(f"{status.capitalize()} in {self.test_info['current_steps'] - 1} steps!")
            logger.info(f"Success rate: {self.test_info['success'] / (self.test_info['success'] + self.test_info['failed'])}. "
                        f"Success trials: {self.test_info['success']}, failed trials: {self.test_info['failed']}. "
                        f"Average steps: {np.mean(self.test_info['history_steps'])}.")
        elif status == 'action':
            logger.info(f"Not successful yet, current steps: {self.test_info['current_steps']}.")
        else:
            raise ValueError(f"Unknown test status: {status}")
        logger.info("-"*40)