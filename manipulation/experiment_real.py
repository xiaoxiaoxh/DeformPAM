import sys
import os
import os.path as osp
import pickle

"""
for test api
"""
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))
sys.path.insert(0, "lib_wrapper/lib")
sys.path.insert(0, "../")
sys.path.insert(0, osp.join(osp.join(osp.dirname(osp.abspath(__file__)), 'third_party')))
sys.path.insert(0, osp.join(osp.join(osp.dirname(osp.abspath(__file__)), 'third_party', 'lib_py')))

from controller.configs.config import config_tshirt_short as planning_config_tshirt_short
from controller.configs.config import config_tshirt_long as planning_config_tshirt_long
from controller.configs.error_config import error_code

from loguru import logger
import numpy as np
from typing import Tuple, Dict, Any, Union, List, Optional
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig, open_dict, ListConfig
from third_party.phoxi.phoxi_camera import PhoXiCamera
from third_party.phoxi.phoxi_camera_web_client import PhoxiCameraWebClient
from third_party.mech_camera.mechmind_camera_wrapper import MechMindCameraWrapper
from third_party.file_camera import FileCamera

import open3d as o3d
from autolab_core import RigidTransform
from common.experiment_base import ExperimentBase, ExperimentRealTransforms, config_completion, convert_dict
from common.datamodels import (ActionMessage, ActionTypeDef, ObjectTypeDef, ObservationMessage,
                               ExceptionMessage, GeneralDualArmExecutionCheckingResult)
from common.space_util import transform_point_cloud
from manipulation.executable_checker import *
from manipulation.grasp_checker import GraspChecker
from functools import partial
import hydra
from hydra.utils import instantiate
from rich.console import Console
import py_cli_interaction

import math
from copy import deepcopy

class ExperimentReal(ExperimentBase):
    option: OmegaConf
    camera: Union[PhoXiCamera, PhoxiCameraWebClient, MechMindCameraWrapper, FileCamera]
    grasp_checker: GraspChecker
    transforms: ExperimentRealTransforms
    # a dictionary to map action types to their corresponding handler functions action_handlers
    action_handlers = {}
    # a dictionary to map action types to their corresponding controller methods
    action_to_controller_method = {}

    def __init__(self, config: Union[Dict, EasyDict, str]):
      
        self.option: DictConfig = config_completion(config)

        # initialize camera and robot
        assert self._init_camera() is None
        if self.option.compat.use_real_robots:
            assert self._init_robot(self.option.planning) is None
        # register action primitive handlers
        self._register_handlers()

    def _init_camera(self) -> Optional[Exception]:
        # instantiate transforms
        transforms = ExperimentRealTransforms(option=self.option)

        camera_cfg = self.option.compat.camera
        if camera_cfg.use_file_camera:
            if 'FileCamera' in camera_cfg.camera_param['_target_']:
                # real file camera
                self.camera = FileCamera(**camera_cfg.camera_param, transforms=transforms)
            else:
                # fake file camera
                self.camera = FileCamera(target_dir="", transforms=transforms)
            self.segmentation_model = None
        else:
            if 'MechMindCameraWrapper' in camera_cfg.camera_param['_target_']:
                # change config_path in camera params into absolute path
                self.option.compat.camera.camera_param.config_path = osp.abspath(osp.join(
                    osp.dirname(__file__), '..', camera_cfg.camera_param.config_path))
                logger.info(f"camera config path: {self.option.compat.camera.camera_param.config_path}")
            elif 'PhoXiCamera' in camera_cfg.camera_param['_target_']  and \
                    'Web' not in camera_cfg.camera_param['_target_']  and \
                    camera_cfg.camera_param.use_external_camera:
                # TODO: test this
                # change external_calibration_path in camera params based on calibration_path
                self.option.compat.camera.camera_param.external_calibration_path = osp.abspath(osp.join(
                    self.option.compat.calibration_path, camera_cfg.camera_param.external_calibration_path
                ))
                logger.info(f"external camera calibration path: "
                            f"{self.option.compat.camera.camera_param.external_calibration_path}")
            # instantiate camera
            self.camera = instantiate(self.option.compat.camera.camera_param)

            # instantiate segmentation model
            from third_party.grounded_sam.grounded_sam import GroundedSAM
            self.segmentation_model = GroundedSAM(**self.option.compat.segmentation)
        
        # use another camera for grasp checker
        grasp_checker_cfg = self.option.compat.grasp_checker
        if grasp_checker_cfg.enable:
            self.grasp_checker = GraspChecker(grasp_checker_cfg)
        else:
            self.grasp_checker = None

        self.transforms = transforms

        return None
    
    def __del__(self):
        self.camera.stop()
        if self.grasp_checker is not None:
            self.grasp_checker.camera.stop()

    def _init_robot(self, config: Union[EasyDict, DictConfig]) -> Optional[Exception]:
        print(config)
        if self.option.compat.object_type in [ObjectTypeDef.to_string(ObjectTypeDef.TSHIRT_SHORT),
                                               ObjectTypeDef.to_string(ObjectTypeDef.TSHIRT_LONG),
                                               ObjectTypeDef.to_string(ObjectTypeDef.NUT),
                                               ObjectTypeDef.to_string(ObjectTypeDef.ROPE)]:
            from controller.controller import Controller
            if self.option.compat.grasp_checker.enable:
                check_grasp = partial(self.grasp_checker.check_grasp, self.transforms.checker_camera_to_world_transform)
            else:
                check_grasp = lambda **kwargs: (True, None)
            self.controller = Controller(cfg=config, check_grasp=check_grasp)
        else:
            raise NotImplementedError
        # make sure the robot type(left or right)
        self.pick_type = None
        return None

    def assign_to_arm(self, pose1: RigidTransform, pose2: RigidTransform) -> Tuple[RigidTransform, RigidTransform]:
        """
        Assign the poses to the left arm and right arm according to the y-axis of the poses
        returns tuple with (left arm, right arm)
        """

        pose_only1 = pose1[0] if isinstance(pose1, tuple) else pose1
        pose_only2 = pose2[0] if isinstance(pose2, tuple) else pose2

        trans1 = pose_only1.translation if pose_only1 is not None else np.zeros(3)
        trans2 = pose_only2.translation if pose_only2 is not None else np.zeros(3)

        if self.option.compat.machine.vertical_assign.enable and \
              abs(trans1[0] - trans2[0]) > self.option.compat.machine.vertical_assign.vertical_limit and \
              abs(trans1[1] - trans2[1]) < self.option.compat.machine.vertical_assign.horizontal_limit:
            if trans1[0] < trans2[0]:
                self.assign_tcp_frame(pose1, pose2)
                return pose1, pose2
            else:
                self.assign_tcp_frame(pose2, pose1)
                return pose2, pose1
        else:
            if trans1[1] > trans2[1]:
                self.assign_tcp_frame(pose1, pose2)
                return pose1, pose2
            else:
                self.assign_tcp_frame(pose2, pose1)
                return pose2, pose1

    def virtual_pose_to_world_pose(self, pose_in_world_space_6d: np.ndarray) -> RigidTransform:
        return self.transforms.virtual_pose_to_world_pose(pose_in_world_space_6d)

    def is_action_executable(self, action_type: ActionTypeDef,
                             poses: Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform],
                             verbose=True,
                             return_detailed_err: bool = False) -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Jude whether the input action with 6D poses is executable,
        return Dict with transforms (3D poses)
        return None, ExceptionMessage if action is not executable
        """
        pick1, pick2, place1, place2 = poses
        # TODO: optimize reachability conditions
        checker_params = deepcopy(self.option.compat.machine.action_checker)
        # overwrite checker params if the action type is in the checker_params
        if ActionTypeDef.to_string(action_type) in checker_params:
            for k, v in checker_params[ActionTypeDef.to_string(action_type)].items():
                checker_params[k] = v
        context = GeneralDualArmExecutionCheckingContext(
                execution_result=GeneralDualArmExecutionCheckingResult(),
                checker_params=checker_params,
                controller=self.controller,
        )
        if action_type == ActionTypeDef.LIFT:
            # for lift, we only use pick1 or pick2
            pick1, pick2 = self.assign_to_arm(pick1, pick2)

            ret_left, ret_right = None, None
            # check left arm
            context.reset()
            context.assign_target_poses(pose_left=pick1)
            checker_left = [partial(single_workspace_checker, is_left_robot=True, is_start=False),
                            partial(single_ik_checker, is_left_robot=True, is_start=False)]
            check_execution_all(context, checker_left)
            context.finish()
            if context.execution_result.overall_success:
                ret_left = deepcopy({'pick_left': pick1, 'joints_value_left_list': context.cached_joints_value_left_list}), None

            context.reset()
            context.assign_target_poses(pose_right=pick2)
            checker_right = [partial(single_workspace_checker, is_left_robot=False, is_start=False),
                            partial(single_ik_checker, is_left_robot=False, is_start=False)]
            check_execution_all(context, checker_right)
            context.finish()
            if context.execution_result.overall_success:
                ret_right =  deepcopy({'pick_right': pick2, 'joints_value_right_list': context.cached_joints_value_right_list}), None

            if ret_left is not None and ret_right is not None:
                # if both arms can lift the object, return the arm with safer joint value
                jacobi_eigenvalue_left = self.controller.checkJacobiSvd("left_robot",
                                                                        np.array(ret_left[0]['joints_value_left_list'][0]),
                                                                        np.array(ret_right[0]['joints_value_right_list'][0]))
                jacobi_eigenvalue_right = self.controller.checkJacobiSvd("right_robot",
                                                                         np.array(ret_left[0]['joints_value_left_list'][0]),
                                                                         np.array(ret_right[0]['joints_value_right_list'][0]))
                condition_number_left = jacobi_eigenvalue_left[0] / (jacobi_eigenvalue_left[-1] + 1e-6)
                condition_number_right = jacobi_eigenvalue_right[0] / (jacobi_eigenvalue_right[-1] + 1e-6)
                if condition_number_left < condition_number_right:
                    return ret_left
                else:
                    return ret_right
            elif ret_left is not None:
                return ret_left
            elif ret_right is not None:
                return ret_right

            # if the action is not executable, return the error message
            err_transform_dict = {'pick_left': pick1, 'pick_right': pick2}
            return (err_transform_dict,
                    ExceptionMessage(f"{ActionTypeDef.to_string(action_type)} action is not executable",
                                     code=context.execution_result.error_types))
        elif action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
            # for sweep, we only use pick1 and place1
            self.assign_tcp_frame(pick1, pick2)
            self.assign_tcp_frame(place1, place2)
            
            context.reset()
            context.assign_target_poses(pose_left=pick1)
            checker = [
                partial(single_workspace_checker, is_left_robot=True, is_start=False),
                partial(single_ik_checker, is_left_robot=True, is_start=False),
                dual_joints_value_checker,
                partial(dual_collision_checker, type=['general', 'between_robots_distance', 'robots_desktop_distance']),
                dual_planning_checker,
                # partial(dual_trajectory_checker, type=['joints']),
            ]
            left_executable = check_execution_all(context, checker)
            context.finish()

            left_transform_dict = {'pick_left': pick1, 'place_left': place1, 'joints_value_left_list': context.cached_joints_value_left_list}
            
            context.reset()
            context.assign_target_poses(pose_right=pick1)
            checker = [
                partial(single_workspace_checker, is_left_robot=False, is_start=False),
                partial(single_ik_checker, is_left_robot=False, is_start=False),
                dual_joints_value_checker,
                partial(dual_collision_checker, type=['general', 'between_robots_distance', 'robots_desktop_distance']),
                dual_planning_checker,
                # partial(dual_trajectory_checker, type=['joints']),
            ]
            right_executable = check_execution_all(context, checker)
            context.finish()

            right_transform_dict = {'pick_right': pick1, 'place_right': place1, 'joints_value_right_list': context.cached_joints_value_right_list}

            if left_executable and not right_executable:
                return left_transform_dict, None
            elif not left_executable and right_executable:
                return right_transform_dict, None
            elif left_executable and right_executable:
                if pick1.translation[1] >= 0:
                    return left_transform_dict, None
                else:
                    return right_transform_dict, None

            checking_result: GeneralDualArmExecutionCheckingResult = context.execution_result
            err_transform_dict = {'pick_left': pick1, 'pick_right': pick2, 'place_left': place1, 'place_right': place2}
            # TODO: return detailed error
            return err_transform_dict, ExceptionMessage("Sweep action is not executable",
                                                        code=checking_result.error_types)
        elif action_type == ActionTypeDef.FLING:
            # for fling, we only use pick1 and pick2
            pick1, pick2 = self.assign_to_arm(pick1, pick2)
            self.assign_tcp_frame(pick1, pick2)
            
            context.reset()
            context.assign_target_poses(pose_left=pick1, pose_right=pick2)
            checker = [
                partial(single_workspace_checker, is_left_robot=True, is_start=False),
                partial(single_workspace_checker, is_left_robot=False, is_start=False),
                partial(dual_ik_joints_value_collision_checker, is_start=False, type=['general', 'between_robots_distance', 'robots_desktop_distance']),
                # dual_planning_checker,
                # partial(dual_trajectory_checker, type=['joints']),
            ]
            check_execution_all(context, checker)
            context.finish()

            # check whether the action can be executed
            checking_result: GeneralDualArmExecutionCheckingResult = context.execution_result
            if checking_result.overall_success:
                # if the action is executable, return the pick points
                return {'pick_left': pick1, 'pick_right': pick2, 'joints_value_left_list': context.cached_joints_value_left_list, 'joints_value_right_list': context.cached_joints_value_right_list}, None
            else:
                # if the action is not executable, return the error message
                err_transform_dict = {'pick_left': pick1, 'pick_right': pick2}
                return err_transform_dict, ExceptionMessage("Fling action is not executable", code=checking_result.error_types)

        elif action_type in {ActionTypeDef.FOLD_1_1, ActionTypeDef.FOLD_1_2, ActionTypeDef.FOLD_2,
                             ActionTypeDef.PICK_AND_PLACE,
                             ActionTypeDef.DRAG,
                             ActionTypeDef.STRAIGHTEN_AND_PLACE}:
            self.assign_tcp_frame(pick1, pick2)
            pick_left, pick_right = pick1, pick2

            self.assign_tcp_frame(place1, place2)
            place_left, place_right = place1, place2

            context.reset()
            context.assign_target_poses(pose_left=pick_left, pose_right=pick_right)
            checker = [
                partial(single_workspace_checker, is_left_robot=True, is_start=False),
                partial(single_workspace_checker, is_left_robot=False, is_start=False),
                partial(dual_ik_joints_value_collision_checker, is_start=False, type=['general', 'between_robots_distance', 'robots_desktop_distance']),
                # dual_planning_checker,
                # partial(dual_trajectory_checker, type=['joints']),
            ]
            check_execution_all(context, checker)
            
            context.tick()
            context.assign_target_poses(pose_left=place_left, pose_right=place_right)
            # checker[-1] = partial(dual_trajectory_checker, type=['joints', 'ends'])
            check_execution_all(context, checker)
            context.finish()

            checking_result: GeneralDualArmExecutionCheckingResult = context.execution_result
            
            if checking_result.overall_success:
                if action_type in (ActionTypeDef.FOLD_2, ActionTypeDef.DRAG):
                    # we only use tcp pose here (no joint values)
                    return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left,
                            'place_right': place_right}, None
                else:
                    # if the action is executable, return the pick points and place points and joint values
                    return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left, 'place_right': place_right,
                            'joints_value_left_list': context.cached_joints_value_left_list,
                            'joints_value_right_list': context.cached_joints_value_right_list}, None
            else:
                # if the action is not executable, return the error message
                err_transform_dict = {'pick_left': pick1, 'pick_right': pick2, 'place_left': place1, 'place_right': place2}
                logger.warning(f"{action_type} action is not executable, "
                             f"error types: {checking_result.error_types},"
                             f"error transform dict: {err_transform_dict}")
                return err_transform_dict, ExceptionMessage(f"{action_type} action is not executable",
                                                            code=checking_result.error_types)
        else:
            # temporary dict for error handlers
            err_transform_dict = {'pick_left': pick1, 'pick_right': pick2, 'place_left': place1, 'place_right': place2}
            return err_transform_dict, ExceptionMessage(f"Unknown action type: {action_type}", code=None)

    def get_pick_points_in_virtual(self, action: ActionMessage) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pick points from world coordinate system to virtual coordinate system
        Return a list (left, right) of 6d vector: (X, Y, Z, R, P, Y)
        """
        pick_left, pick_right = action.pick_points[0], action.pick_points[1]
        if pick_left is None:
            pick_left_numpy = np.zeros((6, )).astype(np.float32)
        else:
            pick_left_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_left)
            pick_left_numpy = np.concatenate([pick_left_in_virtual.translation, pick_left_in_virtual.euler_angles],
                                             axis=-1)

        if pick_right is None:
            pick_right_numpy = np.zeros((6, )).astype(np.float32)
        else:
            pick_right_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_right)
            pick_right_numpy = np.concatenate([pick_right_in_virtual.translation, pick_right_in_virtual.euler_angles], axis=-1)
        return pick_left_numpy, pick_right_numpy

    def get_place_points_in_virtual(self, action: ActionMessage) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pick points from world coordinate system to virtual coordinate system
        Return a list (left, right) of 6d vector: (X, Y, Z, R, P, Y)
        """
        place_left, place_right = action.place_points[0], action.place_points[1]
        if place_left is None:
            place_left_numpy = np.zeros((6, )).astype(np.float32)
        else:
            place_left_in_virtual = self.transforms.world_pose_to_virtual_pose(place_left)
            # TODO: check euler angles (deg or rad?)
            place_left_numpy = np.concatenate([place_left_in_virtual.translation, place_left_in_virtual.euler_angles],
                                              axis=-1)
        if place_right is None:
            place_right_numpy = np.zeros((6, )).astype(np.float32)
        else:
            place_right_in_virtual = self.transforms.world_pose_to_virtual_pose(place_right)
            # TODO: check euler angles (deg or rad?)
            place_right_numpy = np.concatenate([place_right_in_virtual.translation, place_right_in_virtual.euler_angles], axis=-1)
        return place_left_numpy, place_right_numpy

    def get_obs(self) -> Tuple[ObservationMessage, Optional[ExceptionMessage]]:
        """
        Get observation

        Returns:
            Tuple[ObservationMessage, Optional[Exception]]: observation message and exception message
        """
        if type(self.camera) == MechMindCameraWrapper:
            # the pixel indices of RGB image and point cloud are not aligned in MechMind camera
            # so we have to capture RGB image first to calculate mask, then generate masked point cloud with depth map
            # this part takes about 0.6s
            rgb_img = self.camera.capture_rgb()
            camera_pcd = None
        else:
            # PhoXi Camera or FileCamera
            # the pixel indices of RGB image and point cloud are aligned in PhoXi Camera
            # so we can capture RGB image and point cloud simultaneously
            rgb_img, camera_pcd = self.camera.capture_rgb_and_pcd()

        # Filter masks  (this part (SAM model) takes about 0.7s)
        # get masked pcd
        masks = self.segmentation_model.predict(rgb_img)  # (k, h, w)
        mask_sum = masks.sum(axis=-1).sum(axis=-1)  # (k, )
        h, w = masks.shape[1:]
        # filter mask with very large area (probably table)
        logger.debug(f"Mask area ratio: {mask_sum / (h * w)}")
        mask_sum[mask_sum > h * w * self.option.compat.camera.max_mask_area_ratio] = 0
        # TODO: better error handling
        # Exit to avoid collecting invalid data
        if mask_sum.sum() == 0:
            exit("No valid mask detected")
        max_mask_idx = np.argsort(mask_sum)[::-1][0]
        mask_img = np.transpose(masks[max_mask_idx, :, :][np.newaxis, :, :]
                                .repeat(3, axis=0), (1, 2, 0)).astype(np.uint8)  # (h, w, 3)

        if self.option.compat.camera.camera_param.vis:
            import cv2
            cv2.imshow("mask", mask_img*255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if type(self.camera) == MechMindCameraWrapper:
            # get masked pcd with colors (this part takes about 4s)
            camera_pcd, valid_camera_pcd = self.camera.capture_pcd_with_mask(mask_img)
            # transform camera pcd to virtual pcd
            raw_virtual_pcd = camera_pcd.transform(self.transforms.camera_to_virtual_transform)
            raw_virtual_pts = np.asarray(raw_virtual_pcd.points).astype(np.float32)
            # transform valid camera pcd to virtual pcd
            valid_virtual_pcd = valid_camera_pcd.transform(self.transforms.camera_to_virtual_transform)
            valid_virtual_pts = np.asarray(valid_virtual_pcd.points).astype(np.float32)
        else:
            if type(self.camera) == PhoXiCamera and self.camera.use_external_camera:
                # colorize point cloud with RGB image
                pc_xyz = np.asarray(camera_pcd.points).copy()
                # get raw pcd with colors
                pc_rgb = self.camera.colorize_point_cloud(pc_xyz, rgb_img)
                camera_pcd.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)
                # get masked pcd with colors
                pc_mask_color = self.camera.colorize_point_cloud(pc_xyz, mask_img)
                valid_idxs = pc_mask_color[:, 0] > 0
            else:
                mask_reshape = np.reshape(mask_img[:, :, 0], -1)
                valid_idxs = np.where(mask_reshape > 0)

            # transform camera pcd to virtual pcd
            raw_virtual_pcd = camera_pcd.transform(self.transforms.camera_to_virtual_transform)
            raw_virtual_pts = np.asarray(raw_virtual_pcd.points).astype(np.float32)

            # filter out invalid points
            valid_virtual_pcd = o3d.geometry.PointCloud()
            valid_virtual_pcd.points = o3d.utility.Vector3dVector(raw_virtual_pts[valid_idxs])
            valid_virtual_pts = np.asarray(valid_virtual_pcd.points).astype(np.float32)

        if self.option.compat.camera.crop.enable:
            # create axis-aligned bounding box
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.option.compat.camera.crop.min_xyz,
                                                       max_bound=self.option.compat.camera.crop.max_xyz)

            # use crop function to crop point cloud
            valid_virtual_pcd = valid_virtual_pcd.crop(aabb)
            valid_virtual_pts = np.asarray(valid_virtual_pcd.points).astype(np.float32)

        res = ObservationMessage(valid_virtual_pts, valid_virtual_pcd, raw_virtual_pts, raw_virtual_pcd,
                                 mask_img, rgb_img)

        if self.option.compat.camera.camera_param.vis:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            logger.debug(f"valid_virtual_pts.shape: {valid_virtual_pts.shape}")
            o3d.visualization.draw_geometries([coord, valid_virtual_pcd])

        return res, None
    
    def mark_pcd_with_workspace(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd_xyz_points = transform_point_cloud(np.asarray(pcd.points), self.transforms.virtual_to_world_transform)
        pcd_colors = np.asarray(pcd.colors)
        
        pcd_colors *= 0.5
        x_lim = self.option.compat.machine.x_lim_m
        y_lim = self.option.compat.machine.y_lim_m
        pcd_in_workspace = (pcd_xyz_points[:, 0] > x_lim[0]) & (pcd_xyz_points[:, 0] < x_lim[1]) & \
                            (pcd_xyz_points[:, 1] > y_lim[0]) & (pcd_xyz_points[:, 1] < y_lim[1])
        left_workspace_min_x = self.option.compat.machine.left_workspace_min_x
        left_workspace_y_limits = self.option.compat.machine.left_workspace_y_limits
        left_range_idx = np.where(pcd_in_workspace &
                                (pcd_xyz_points[:, 0] > left_workspace_min_x) &
                                (pcd_xyz_points[:, 1] > left_workspace_y_limits[0]) &
                                (pcd_xyz_points[:, 1] < left_workspace_y_limits[1]))[0]
        pcd_colors[left_range_idx, 1] *= 2
        
        right_workspace_min_x = self.option.compat.machine.right_workspace_min_x
        right_workspace_y_limits = self.option.compat.machine.right_workspace_y_limits
        right_range_idx = np.where(pcd_in_workspace &
                                    (pcd_xyz_points[:, 0] > right_workspace_min_x) &
                                    (pcd_xyz_points[:, 1] > right_workspace_y_limits[0]) &
                                        (pcd_xyz_points[:, 1] < right_workspace_y_limits[1]))[0]
        pcd_colors[right_range_idx, 2] *= 2
        
        wiper_width = self.option.compat.machine.wiper_width
        if wiper_width is not None:
            line_width = wiper_width / 20
            min_x = max(x_lim[0], left_workspace_min_x, right_workspace_min_x)
            max_x = x_lim[1]
            min_y = max(y_lim[0], min(left_workspace_y_limits[0], right_workspace_y_limits[0]))
            max_y = min(y_lim[1], max(left_workspace_y_limits[1], right_workspace_y_limits[1]))
            for h in np.arange(min_x, max_x, wiper_width):
                pcd_colors[np.where((pcd_xyz_points[:, 0] > h - line_width) & (pcd_xyz_points[:, 0] < h + line_width))[0], :] = 0.1
            for w in np.arange(min_y, max_y, wiper_width):
                pcd_colors[np.where((pcd_xyz_points[:, 1] > w - line_width) & (pcd_xyz_points[:, 1] < w + line_width))[0], :] = 0.1
                            
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        
        return pcd
        

    def is_object_reachable(self, mask: np.ndarray) -> bool:
        """
        Check whether the object is reachable according to mask information

        @params: mask (H, W, 3) uint8, np.ndarray
        """
        limits = self.option.compat.machine.image_width_ratio_limits_for_reachability_det
        mask_single = mask[:, :, 0]
        h, w = mask_single.shape[:2]
        valid_idxs = np.nonzero(mask_single)
        # check whether the object center is in the center region of the table
        return w * limits[0] <= valid_idxs[1].mean() <= w * limits[1]

    def is_object_on_table(self, mask: np.ndarray) -> bool:
        """
        Check whether the object is still on the table according to mask information

        @params: mask (H, W, 3) uint8, np.ndarray
        """
        mask_single = mask[:, :, 0]
        mask_sum = mask_single.sum(axis=-1).sum(axis=-1)  # (k, )
        h, w = mask_single.shape
        # if the mask area is too large, it probably covers the whole table,
        # and in this case the object is probably not on the table
        return mask_sum <= h * w * self.option.compat.camera.max_mask_area_ratio

    @classmethod
    def register_action_handler(cls, action_type, controller_method_name=None):
        """Define a decorator function to register handler functions or controller methods"""
        def decorator(func):
            # register action handler
            cls.action_handlers[action_type] = func
            return func

        if controller_method_name:
            # register controller method and template handler function if controller_method_name is provided
            cls.action_to_controller_method[action_type] = controller_method_name
            return decorator(cls._template_handler)
        else:
            # register handler function if controller_method_name is not provided
            return decorator

    @classmethod
    def _register_handlers(cls):
        """Register action handlers for different action types"""
        cls.register_action_handler(ActionTypeDef.HOME)(cls._handle_home)
        cls.register_action_handler(ActionTypeDef.DONE)(cls._handle_done)
        cls.register_action_handler(ActionTypeDef.FLING)(cls._handle_fling)
        cls.register_action_handler(ActionTypeDef.DRAG, 'execute_drag_complaint')
        # TODO: implement drag_hybrid action
        cls.register_action_handler(ActionTypeDef.FOLD_1_1, 'execute_fold_one')
        cls.register_action_handler(ActionTypeDef.FOLD_1_2, 'execute_fold_one')
        cls.register_action_handler(ActionTypeDef.FOLD_2, 'execute_fold_two')
        # TODO: implement pick_and_place action
        cls.register_action_handler(ActionTypeDef.LIFT)(cls._handle_lift)
        cls.register_action_handler(ActionTypeDef.GRASP_WIPER)(cls._handle_grasp_wiper)
        cls.register_action_handler(ActionTypeDef.SWEEP)(cls._handle_sweep)
        cls.register_action_handler(ActionTypeDef.STRAIGHTEN_AND_PLACE)(cls._handle_straighten_and_place)
        cls.register_action_handler(ActionTypeDef.SINGLE_PICK_AND_PLACE)(cls._handle_single_pick_and_place)
        cls.register_action_handler(ActionTypeDef.MANUALLY_RANDOMIZE)(cls._handle_manually_randomize)

    @staticmethod
    def _default_interface_conversion(action: ActionMessage) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Default interface conversion logic for action message to controller input"""
        if action.pick_points[0] is not None:
            pick_left_trans, pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
            pick_left = np.concatenate([pick_left_trans, pick_left_quaterion]).tolist()
        else:
            pick_left = None

        if action.pick_points[1] is not None:
            pick_right_trans, pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
            pick_right = np.concatenate([pick_right_trans, pick_right_quaterion]).tolist()
        else:
            pick_right = None

        if action.place_points[0] is not None:
            place_left_trans, place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion
            place_left = np.concatenate([place_left_trans, place_left_quaterion]).tolist()
        else:
            place_left = None

        if action.place_points[1] is not None:
            place_right_trans, place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
            place_right = np.concatenate([place_right_trans, place_right_quaterion]).tolist()
        else:
            place_right = None

        joints_value_list = action.joints_value_list
        extra_params = action.extra_action_params

        return pick_left, pick_right, place_left, place_right, joints_value_list, extra_params

    def _template_handler(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left, pick_right, place_left, place_right, joints_value_list, extra_params = (
            self._default_interface_conversion(action))
        method_name = self.action_to_controller_method[action.action_type]
        controller_method = getattr(self.controller, method_name)
        if joints_value_list[0] is None or joints_value_list[1] is None:
            logger.debug(f"use poses for {action.action_type}")
            logger.debug(f"{action.action_type} poses:    pick_left:{pick_left}, pick_right:{pick_right}, "
                     f"place_left:{place_left}, place_right:{place_right}")
            execution_result = controller_method(np.array([pick_left, pick_right, place_left, place_right]),
                                                 **extra_params)
        else:
            logger.debug(f"use joints values for {action.action_type}")
            logger.debug(f"{action.action_type} joints values:    pick_left:{joints_value_list[0][0]}, pick_right:{joints_value_list[1][0]}, "
                        f"place_left:{joints_value_list[0][1]}, place_right:{joints_value_list[1][1]}")
            execution_result = controller_method(np.array([joints_value_list[0][0], joints_value_list[1][0],
                                                           joints_value_list[0][1], joints_value_list[1][1]]),
                                                 joint_enable=True,
                                                 **extra_params)
        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def _handle_home(self, _: ActionMessage) -> Optional[ExceptionMessage]:
        self.controller.goToHome()
        return None

    def _handle_done(self, _: ActionMessage) -> Optional[ExceptionMessage]:
        self.controller.goToHome()
        return None

    def _handle_manually_randomize(self, _: ActionMessage) -> Optional[ExceptionMessage]:
        while True:
            finished = py_cli_interaction.must_parse_cli_bool('Is the object manually randomized?',
                                                              default_value=True)
            if finished:
                break
        return None

    def _handle_fling(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left, pick_right, place_left, place_right, joints_value_list, extra_params = (
            self._default_interface_conversion(action))
        if joints_value_list[0] is None or joints_value_list[1] is None:
            poses = np.array([pick_left, pick_right])
            logger.debug(f"use poses for fling")
            logger.debug(f"fling action poses:    pick_left:{pick_left}, pick_right:{pick_right}")
            execution_result = self.controller.execute_dual_fling(poses)
        else:
            poses = np.array([joints_value_list[0][0], joints_value_list[1][0]])
            logger.debug(f"use joints values for fling")
            logger.debug(f"fling action joints values:    pick_left:{joints_value_list[0][0]}, pick_right:{joints_value_list[1][0]}")
            execution_result = self.controller.execute_dual_fling(poses, joint_enable=True)

        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def _handle_lift(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.pick_points[0] is not None:
            pick_trans, pick_quat = (action.pick_points[0].translation, action.pick_points[0].quaternion)
            pick_joint_value = action.joints_value_list[0]
            if pick_joint_value is None:
                pick_pose = np.concatenate([pick_trans, pick_quat]).tolist()
                logger.debug(f"lift action pose:    pick_pose:{pick_pose}")
                execution_result = self.controller.execute_lift(pose=pick_pose, robot="l", joint_enable=False)
            else:
                logger.debug(f"lift action joints value:    pick_joint_value:{pick_joint_value[0]}")
                execution_result = self.controller.execute_lift(pose=pick_joint_value[0], robot="l", joint_enable=True)
        elif action.pick_points[1] is not None:
            pick_trans, pick_quat = action.pick_points[1].translation, action.pick_points[1].quaternion
            pick_joint_value = action.joints_value_list[1]
            if pick_joint_value is None:
                pick_pose = np.concatenate([pick_trans, pick_quat]).tolist()
                logger.debug(f"lift action pose:    pick_pose:{pick_pose}")
                execution_result = self.controller.execute_lift(pose=pick_pose, robot="r", joint_enable=False)
            else:
                logger.debug(f"lift action joints value:    pick_joint_value:{pick_joint_value[0]}")
                execution_result = self.controller.execute_lift(pose=pick_joint_value[0], robot="r", joint_enable=True)
        else:
            logger.exception("No pick point is provided for lift action")
            return ExceptionMessage("No pick point is provided for lift action", code=None)

        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def _handle_grasp_wiper(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        for robot_type in ["l", "r"]:
            grasped = False
            while not grasped:
                self.controller.actuator.open_gripper(robot=robot_type)
                Console().print(f"[instruction] Grasp wiper with {robot_type} robot")
                py_cli_interaction.must_parse_cli_bool("Start grasping?", default_value=True)
                self.controller.actuator.close_gripper(robot=robot_type)
                grasped = py_cli_interaction.must_parse_cli_bool("Grasped?", default_value=False)
        return None

    def _handle_sweep(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.pick_points[0] is not None:
            robot_type = 'left_robot'
            logger.debug(f"sweep action:    robot type:{robot_type}")
            start_trans, start_quat = (action.pick_points[0].translation, action.pick_points[0].quaternion)
            pick_joint_value = action.joints_value_list[0]
            if pick_joint_value is None:
                joint_enable = False
                start_pose = np.concatenate([start_trans, start_quat]).tolist()
                logger.debug(f"sweep poses:    start:{start_pose}")
            else:
                joint_enable =  True
                start_pose = pick_joint_value[0]
                logger.debug(f"sweep joints values:    start:{start_pose}")
            end_trans, end_quat = (action.place_points[0].translation, action.place_points[0].quaternion)
            end_pose = np.concatenate([end_trans, end_quat]).tolist()
            logger.debug(f"sweep poses:    end:{end_pose}")
            poses = np.array([start_pose, end_pose])
        elif action.pick_points[1] is not None:
            robot_type = 'right_robot'
            logger.debug(f"sweep action:    robot type:{robot_type}")
            start_trans, start_quat = (action.pick_points[1].translation, action.pick_points[1].quaternion)
            pick_joint_value = action.joints_value_list[1]
            if pick_joint_value is None:
                joint_enable = False
                start_pose = np.concatenate([start_trans, start_quat]).tolist()
                logger.debug(f"sweep poses:    start:{start_pose}")
            else:
                joint_enable = True
                start_pose = pick_joint_value[0]
                logger.debug(f"sweep joints values:    start:{start_pose}")
            end_trans, end_quat = (action.place_points[1].translation, action.place_points[1].quaternion)
            end_pose = np.concatenate([end_trans, end_quat]).tolist()
            logger.debug(f"sweep poses:    end:{end_pose}")
            poses = np.array([start_pose, end_pose])

        execution_result = self.controller.execute_sweep(poses, robot_type, joint_enable=joint_enable)

        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def _handle_straighten_and_place(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left, pick_right, place_left, place_right, joints_value_list, extra_params = (
            self._default_interface_conversion(action))
        if joints_value_list[0] is None or joints_value_list[1] is None:
            logger.debug(f"use poses for {action.action_type}")
            logger.debug(f"{action.action_type} poses:    pick_left:{pick_left}, pick_right:{pick_right}, "
                     f"place_left:{place_left}, place_right:{place_right}")
            execution_result = self.controller.execute_straighten_and_place(
                                                np.array([pick_left, pick_right, place_left, place_right]),
                                                 **extra_params)
        else:
            logger.debug(f"use joints values for {action.action_type}")
            logger.debug(f"{action.action_type} joints values:    pick_left:{joints_value_list[0][0]}, pick_right:{joints_value_list[1][0]}, "
                        f"place_left:{joints_value_list[0][1]}, place_right:{joints_value_list[1][1]}")
            # only use the begin joints values
            execution_result = self.controller.execute_straighten_and_place(
                                                np.array([joints_value_list[0][0], joints_value_list[1][0],
                                                place_left, place_right]),
                                                 joint_enable=True,
                                                 **extra_params)
        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def _handle_single_pick_and_place(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.pick_points[0] is not None:
            robot_type = 'left_robot'
            logger.debug(f"single_pick_and_place action:    robot type:{robot_type}")
            start_trans, start_quat = (action.pick_points[0].translation, action.pick_points[0].quaternion)
            pick_joint_value = action.joints_value_list[0]
            if pick_joint_value is None:
                joint_enable = False
                start_pose = np.concatenate([start_trans, start_quat]).tolist()
                logger.debug(f"single_pick_and_place poses:    start:{start_pose}")
            else:
                joint_enable =  True
                start_pose = pick_joint_value[0]
                logger.debug(f"single_pick_and_place joints values:    start:{start_pose}")
            end_trans, end_quat = (action.place_points[0].translation, action.place_points[0].quaternion)
            end_pose = np.concatenate([end_trans, end_quat]).tolist()
            logger.debug(f"single_pick_and_place poses:    end:{end_pose}")
            poses = np.array([start_pose, end_pose])
            high = np.linalg.norm(action.pick_points[0].translation[:2] - action.place_points[0].translation[:2])
        elif action.pick_points[1] is not None:
            robot_type = 'right_robot'
            logger.debug(f"single_pick_and_place action:    robot type:{robot_type}")
            start_trans, start_quat = (action.pick_points[1].translation, action.pick_points[1].quaternion)
            pick_joint_value = action.joints_value_list[1]
            if pick_joint_value is None:
                joint_enable = False
                start_pose = np.concatenate([start_trans, start_quat]).tolist()
                logger.debug(f"single_pick_and_place poses:    start:{start_pose}")
            else:
                joint_enable = True
                start_pose = pick_joint_value[0]
                logger.debug(f"single_pick_and_place joints values:    start:{start_pose}")
            end_trans, end_quat = (action.place_points[1].translation, action.place_points[1].quaternion)
            end_pose = np.concatenate([end_trans, end_quat]).tolist()
            logger.debug(f"single_pick_and_place poses:    end:{end_pose}")
            poses = np.array([start_pose, end_pose])
            high = np.linalg.norm(action.pick_points[1].translation[:2] - action.place_points[1].translation[:2])

        execution_result = self.controller.execute_single_pick_and_place(poses, high, robot_type, joint_enable=joint_enable)

        if execution_result == error_code.ok:
            return None
        else:
            return ExceptionMessage(f"Error type {execution_result} when executing "
                                    f"{ActionTypeDef.to_string(action.action_type)} action",
                                    code=execution_result)

    def execute_action(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        # when not using real robots, return None
        if not self.option.compat.use_real_robots:
            return None

        # look up the handler function in the dictionary using the action type
        handler = self.action_handlers.get(action.action_type)
        if handler is None:
            return ExceptionMessage(f"Unknown action type: {action.action_type}", code=None)
        # call the handler function with the action
        return handler(self, action)
    
    def move_rot_table(self, degrees=None):
        self.controller.move_rot_table(degrees)


@hydra.main(
    config_path="../config/supervised_experiment", config_name="experiment_supervised_base.yaml", version_base="1.1"
)
def main(cfg: DictConfig):
    # use virtual camera (file camera)
    cfg.experiment.compat.camera.use_file_camera=True

    if cfg.experiment.compat.object_type == 'tshirt_long':
        planning_config = planning_config_tshirt_long
    elif cfg.experiment.compat.object_type == 'tshirt_short':
        planning_config = planning_config_tshirt_short
    else:
        raise NotImplementedError
    cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))
    # create experiment
    exp = ExperimentReal(config=cfg.experiment)



if __name__ == '__main__':
    main()