from dataclasses import dataclass, field
import json
from abc import ABC, abstractmethod
from os import path as osp
from typing import Tuple, Dict, Optional, Union

import numpy as np
import open3d as o3d
from autolab_core import RigidTransform, transformations as tr
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict
from scipy.spatial.transform import Rotation

from common.datamodels import (ActionTypeDef, ObjectTypeDef, ActionMessage,
                               ObservationMessage, ExceptionMessage, GeneralDualArmExecutionCheckingResult)

from loguru import logger
from copy import deepcopy
import math

class ExperimentBase(ABC):
    option: OmegaConf

    @abstractmethod
    def __del__(self):
        pass

    @staticmethod
    def assign_tcp_frame(pose_left: RigidTransform, pose_right: RigidTransform):
        if isinstance(pose_left, tuple):
            for p in pose_left:
                p.from_frame = 'l_tcp'
        elif pose_left is not None:
            pose_left.from_frame = 'l_tcp'

        if isinstance(pose_right, tuple):
            for p in pose_right:
                p.from_frame = 'r_tcp'
        elif pose_right is not None:
            pose_right.from_frame = 'r_tcp'

    @abstractmethod
    def assign_to_arm(self, pose1: RigidTransform, pose2: RigidTransform) -> Tuple[RigidTransform, RigidTransform]:
        """returns tuple with (left arm, right arm)"""
        pass

    @abstractmethod
    def is_action_executable(self, action_type: ActionTypeDef,
                             poses: Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform],
                             **kwargs) -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Jude whether the input action with 6D poses is executable,
        return (Dict, None) with transforms (3D poses) if action is valid,
        return (None, ExceptionMessage) if action is not executable
        """
        pass

    @abstractmethod
    def get_obs(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        """
        Get observation from the camera and the segmentation model

        Returns:
            Tuple[ObservationMessage, Optional[Exception]]: observation message and exception message
        """
        pass

    @abstractmethod
    def execute_action(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pass


@dataclass
class ExperimentRealTransforms:
    option: DictConfig = field(default=None)
    world_to_camera_transform: np.ndarray = field(default=np.eye(4))
    camera_to_world_transform: np.ndarray = field(default=np.eye(4))
    world_to_checker_camera_transform: np.ndarray = field(default=np.eye(4))
    checker_camera_to_world_transform: np.ndarray = field(default=np.eye(4))
    world_to_left_robot_transform: np.ndarray = field(default=np.eye(4))
    world_to_right_robot_transform: np.ndarray = field(default=np.eye(4))
    left_robot_to_world_transform: np.ndarray = field(default=np.eye(4))
    right_robot_to_world_transform: np.ndarray = field(default=np.eye(4))
    left_robot_base_pos: np.ndarray = field(default=np.array([0.0, 0.0, 0.0]))
    right_robot_base_pos: np.ndarray = field(default=np.array([0.0, 0.0, 0.0]))
    virtual_to_world_transform: np.ndarray = field(default=np.eye(4))
    world_to_virtual_transform: np.ndarray = field(default=np.eye(4))
    virtual_to_camera_transform: np.ndarray = field(default=np.eye(4))
    camera_to_virtual_transform: np.ndarray = field(default=np.eye(4))
    virtual_intrinsic: o3d.camera.PinholeCameraIntrinsic = field(default=None)

    def __post_init__(self):
        if self.option is not None:
            with open(osp.join(self.option.compat.calibration_path, 'world_to_camera_transform.json'), 'r') as f:
                self.world_to_camera_transform = np.array(json.load(f))
                self.camera_to_world_transform = np.linalg.inv(self.world_to_camera_transform)
            with open(osp.join(self.option.compat.calibration_path, 'world_to_checker_camera_transform.json'), 'r') as f:
                self.world_to_checker_camera_transform = np.array(json.load(f))
                self.checker_camera_to_world_transform = np.linalg.inv(self.world_to_checker_camera_transform)
            with open(osp.join(self.option.compat.calibration_path, 'world_to_left_robot_transform.json'), 'r') as f:
                self.world_to_left_robot_transform = np.array(json.load(f))
            with open(osp.join(self.option.compat.calibration_path, 'world_to_right_robot_transform.json'), 'r') as f:
                self.world_to_right_robot_transform = np.array(json.load(f))
            with open(osp.join(self.option.compat.calibration_path, 'world_to_virtual_transform.json'), 'r') as f:
                self.world_to_virtual_transform = np.array(json.load(f))

            self.left_robot_to_world_transform = np.linalg.inv(self.world_to_left_robot_transform)
            self.right_robot_to_world_transform = np.linalg.inv(self.world_to_right_robot_transform)

            self.left_robot_base_pos = (self.left_robot_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]
            self.right_robot_base_pos = (self.right_robot_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]
            self.virtual_to_world_transform = np.linalg.inv(self.world_to_virtual_transform)

            self.virtual_to_camera_transform = self.world_to_camera_transform @ self.virtual_to_world_transform
            self.camera_to_virtual_transform = np.linalg.inv(self.virtual_to_camera_transform)

            self.virtual_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                osp.join(self.option.compat.calibration_path, 'camera_intrinsic_scanner.json'))
            self.virtual_extrinsic_dummy = np.array([[1., 0., 0., 0.],
                                                     [0., -1., 0., 0.8],
                                                     [0., 0., -1., 1.6],
                                                     [0., 0., 0., 1.]])
        else:
            pass

    def virtual_pose_to_world_pose(self, pose_6d: np.ndarray, euler_type: str = 'XZX') -> RigidTransform:
        xyz = pose_6d[np.newaxis, :3]
        virtual_pcd = o3d.geometry.PointCloud()
        virtual_pcd.points = o3d.utility.Vector3dVector(xyz)
        world_pcd = virtual_pcd.transform(self.virtual_to_world_transform)
        xyz_world = np.asarray(world_pcd.points)[0]
        r, p, y = pose_6d[-3], pose_6d[-2], pose_6d[-1]
        rotation = Rotation.from_euler(euler_type, [r, p, y], degrees=False)
        res = RigidTransform(translation=xyz_world, rotation=rotation.as_matrix(), from_frame='robot_tcp')
        return res

    def world_pose_to_virtual_pose(self, pose_world: RigidTransform) -> RigidTransform:
        """
        pose_world: RigidTransform
        """
        rotation, translation = RigidTransform.rotation_and_translation_from_matrix(self.world_to_virtual_transform)
        world_to_virtual_transform = RigidTransform(rotation=rotation,
                                                    translation=translation,
                                                    from_frame='world',
                                                    to_frame='virtual')
        # res = pose_world * world_to_virtual_transform
        res = world_to_virtual_transform * pose_world
        return res


def convert_dict(config: dict):
    new_dict = dict()
    for key, edict_item in config.items():
        if isinstance(edict_item, dict):
            new_dict[key] = convert_dict(config[key])
        else:
            if isinstance(edict_item, np.ndarray):
                new_dict[key] = edict_item.tolist()
            else:
                new_dict[key] = edict_item
    return new_dict


def config_completion(config: Union[Dict, EasyDict, DictConfig, str]) -> Union[DictConfig, ListConfig]:
    if isinstance(config, str):
        option = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = convert_dict(config)
        option = OmegaConf.create(config)
    elif isinstance(config, EasyDict):
        config = convert_dict(dict(config))
        option = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        option = config
    else:
        raise NotImplementedError

    # automatically override robot positions by reading calibration files
    # option.compat.object_type = ObjectTypeDef.from_string(option.compat.object_type)
    with open(osp.join(option.compat.calibration_path, 'world_to_left_robot_transform.json'), 'r') as f:
        left_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))
    with open(osp.join(option.compat.calibration_path, 'world_to_right_robot_transform.json'), 'r') as f:
        right_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))

    left_rpy_in_world = Rotation.from_matrix(left_robot_to_world_transform[:3, :3]).as_euler('xyz')
    right_rpy_in_world = Rotation.from_matrix(right_robot_to_world_transform[:3, :3]).as_euler('xyz')

    # print("====================[ DEBUG ]====================")
    # print(option.planning.robot_init_positions, [tuple(left_robot_to_world_transform[:3,3].tolist()), tuple(right_robot_to_world_transform[:3,3].tolist())])
    # print(option.planning.robot_init_orientations, [tuple(left_rpy_in_world),tuple(right_rpy_in_world)])
    # print("====================[  END  ]====================")
    option.planning.robot_init_positions = [tuple(left_robot_to_world_transform[:3, 3].tolist()), tuple(right_robot_to_world_transform[:3, 3].tolist())]
    option.planning.robot_init_orientations = [tuple(left_rpy_in_world.tolist()), tuple(right_rpy_in_world.tolist())]
    
    if getattr(option.compat.machine, "action_checker", None) is not None:
        checker_joints_value_limit = option.compat.machine.action_checker.joints_value_limit
        left_joint_limit_lowers = deepcopy(option.planning.motion.left_joint_limit_lowers)
        right_joint_limit_lowers = deepcopy(option.planning.motion.right_joint_limit_lowers)
        left_joint_limit_uppers = deepcopy(option.planning.motion.left_joint_limit_uppers)
        right_joint_limit_uppers = deepcopy(option.planning.motion.right_joint_limit_uppers)
        for joint_idx, limit_list in checker_joints_value_limit.items():
            joint_idx = int(joint_idx)
            limit_value = np.array(limit_list).flatten()
            lower = math.radians(np.min(limit_value))
            upper = math.radians(np.max(limit_value))
            left_joint_limit_lowers[joint_idx] = lower
            right_joint_limit_lowers[joint_idx] = lower
            left_joint_limit_uppers[joint_idx] = upper
            right_joint_limit_uppers[joint_idx] = upper
            logger.info(f"overwrite joint {joint_idx} limit -> {lower} ~ {upper}")
    option.planning.motion.left_joint_limit_lowers = left_joint_limit_lowers
    option.planning.motion.right_joint_limit_lowers = right_joint_limit_lowers
    option.planning.motion.left_joint_limit_uppers = left_joint_limit_uppers
    option.planning.motion.right_joint_limit_uppers = right_joint_limit_uppers
    
    # for rfcontroller
    if getattr(config, 'planning', None) is not None and \
        getattr(config.planning, 'world_to_left_matrix', None) is not None:
            config.planning.world_to_left_matrix = np.linalg.inv(left_robot_to_world_transform).tolist()
            config.planning.world_to_right_matrix = np.linalg.inv(right_robot_to_world_transform).tolist()
            option.planning.world_to_left_matrix = config.planning.world_to_left_matrix
            option.planning.world_to_right_matrix = config.planning.world_to_right_matrix
    
    return option
