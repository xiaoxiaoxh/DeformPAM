from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Iterable
from typing import List, Optional, Tuple

import numpy as np
import omegaconf
import open3d as o3d
from autolab_core import RigidTransform
from rich.console import Console

from copy import deepcopy

# from controller.atom_controller import AtomController

class ActionTypeDef(Enum):
    FAIL = -1
    FLING = 0
    DRAG = 1
    FOLD_1_1 = 2
    FOLD_1_2 = 8
    FOLD_2 = 3
    PICK_AND_PLACE = 4
    DONE = 5
    LIFT = 6
    DRAG_HYBRID = 7  # contact dragging
    MANUALLY_RANDOMIZE = 12
    GRASP_WIPER = 13
    SWEEP = 14
    STRAIGHTEN_AND_PLACE = 15
    SINGLE_PICK_AND_PLACE = 16
    HOME = 100

    @staticmethod
    def from_string(type_str):
        if type_str == 'done':
            return ActionTypeDef.DONE
        elif type_str == 'fail':
            return ActionTypeDef.FAIL
        elif type_str == 'fling':
            return ActionTypeDef.FLING
        elif type_str == 'drag':
            return ActionTypeDef.DRAG
        elif type_str == 'pick_and_place':
            return ActionTypeDef.PICK_AND_PLACE
        elif type_str == 'fold1_1':
            return ActionTypeDef.FOLD_1_1
        elif type_str == 'fold1_2':
            return ActionTypeDef.FOLD_1_2
        elif type_str == 'fold2':
            return ActionTypeDef.FOLD_2
        elif type_str == 'lift':
            return ActionTypeDef.LIFT
        elif type_str == 'drag_hybrid':
            return ActionTypeDef.DRAG_HYBRID
        elif type_str == 'manually_randomize':
            return ActionTypeDef.MANUALLY_RANDOMIZE
        elif type_str == 'grasp_wiper':
            return ActionTypeDef.GRASP_WIPER
        elif type_str == 'sweep':
            return ActionTypeDef.SWEEP
        elif type_str == 'straighten_and_place':
            return ActionTypeDef.STRAIGHTEN_AND_PLACE
        elif type_str == 'single_pick_and_place':
            return ActionTypeDef.SINGLE_PICK_AND_PLACE
        elif type_str == 'none' or type_str == 'None' or type_str == 'null' or type_str is None:
            return None
        else:
            raise NotImplementedError

    @staticmethod
    def to_string(action_type):
        if action_type == ActionTypeDef.DONE:
            return 'done'
        elif action_type == ActionTypeDef.FAIL:
            return 'fail'
        elif action_type == ActionTypeDef.FLING:
            return 'fling'
        elif action_type == ActionTypeDef.DRAG:
            return 'drag'
        elif action_type == ActionTypeDef.PICK_AND_PLACE:
            return 'pick_and_place'
        elif action_type == ActionTypeDef.FOLD_1_1:
            return 'fold1_1'
        elif action_type == ActionTypeDef.FOLD_1_2:
            return 'fold1_2'
        elif action_type == ActionTypeDef.FOLD_2:
            return 'fold2'
        elif action_type == ActionTypeDef.LIFT:
            return 'lift'
        elif action_type == ActionTypeDef.DRAG_HYBRID:
            return 'drag_hybrid'
        elif action_type == ActionTypeDef.MANUALLY_RANDOMIZE:
            return 'manually_randomize'
        elif action_type == ActionTypeDef.GRASP_WIPER:
            return 'grasp_wiper'
        elif action_type == ActionTypeDef.SWEEP:
            return 'sweep'
        elif action_type == ActionTypeDef.STRAIGHTEN_AND_PLACE:
            return 'straighten_and_place'
        elif action_type == ActionTypeDef.SINGLE_PICK_AND_PLACE:
            return 'single_pick_and_place'
        elif action_type is None:
            return 'null'
        else:
            raise NotImplementedError(f'unknown action type {action_type}')


class ObjectTypeDef(Enum):
    NONE = 0
    TSHIRT_SHORT = 1
    TSHIRT_LONG = 2
    NUT = 3
    ROPE = 4

    @staticmethod
    def from_string(type_str):
        if type_str == 'tshirt_short':
            return ObjectTypeDef.TSHIRT_SHORT
        elif type_str == 'tshirt_long':
            return ObjectTypeDef.TSHIRT_LONG
        elif type_str == 'nut':
            return ObjectTypeDef.NUT
        elif type_str == 'rope':
            return ObjectTypeDef.ROPE
        else:
            raise NotImplementedError

    @staticmethod
    def to_string(object_type):
        if object_type == ObjectTypeDef.TSHIRT_SHORT:
            return 'tshirt_short'
        elif object_type == ObjectTypeDef.TSHIRT_LONG:
            return 'tshirt_long'
        elif object_type == ObjectTypeDef.NUT:
            return 'nut'
        elif object_type == ObjectTypeDef.ROPE:
            return 'rope'
        else:
            raise NotImplementedError


class GeneralExecutionCheckingErrorType(Enum):
    """
    An enum class to store the types of general errors that can occur during execution checking.
    """
    NO_ERROR = auto()
    LEFT_ARM_OUT_OF_WORKSPACE = auto()
    RIGHT_ARM_OUT_OF_WORKSPACE = auto()
    LEFT_ARM_NO_IK = auto()
    RIGHT_ARM_NO_IK = auto()
    DUAL_ARM_NO_IK = auto()
    LEFT_ARM_JOINTS_VALUE_ERROR = auto()
    RIGHT_ARM_JOINTS_VALUE_ERROR = auto()
    LEFT_ARM_JACOBIAN_ERROR = auto()
    RIGHT_ARM_JACOBIAN_ERROR = auto()
    COLLISION_DETECTED = auto()
    DUAL_ARM_NO_PLANNING = auto()
    LEFT_ARM_TRAJECTORY_ERROR = auto()
    RIGHT_ARM_TRAJECTORY_ERROR = auto()


@dataclass
class GeneralDualArmExecutionCheckingResult:
    """
    A general dataclass to store the result of checking the execution of given target TCP poses.
    """
    left_arm_within_workspace: bool = field(default=True, init=True)
    right_arm_within_workspace: bool = field(default=True, init=True)
    left_arm_ik_valid: bool = field(default=True, init=True)
    right_arm_ik_valid: bool = field(default=True, init=True)
    dual_arm_ik_valid: bool = field(default=True, init=True)
    left_arm_joints_value_valid: bool = field(default=True, init=True)
    right_arm_joints_value_valid: bool = field(default=True, init=True)
    left_arm_jacobian_valid: bool = field(default=True, init=True)
    right_arm_jacobian_valid: bool = field(default=True, init=True)
    no_collision: bool = field(default=True, init=True)  # we set this to True by default because we assume no collision
    dual_arm_planning_valid: bool = field(default=True, init=True)  # we set this to True by default because we assume planning is valid
    left_arm_trajectory_valid: bool = field(default=True, init=True)  # we set this to True by default because we assume trajectory is valid
    right_arm_trajectory_valid: bool = field(default=True, init=True)  # we set this to True by default because we assume trajectory is valid
            
    @property
    def error_types(self):
        error_types = []
        if not self.left_arm_within_workspace:
            error_types.append(GeneralExecutionCheckingErrorType.LEFT_ARM_OUT_OF_WORKSPACE)
        if not self.right_arm_within_workspace:
            error_types.append(GeneralExecutionCheckingErrorType.RIGHT_ARM_OUT_OF_WORKSPACE)
        if not self.left_arm_ik_valid:
            error_types.append(GeneralExecutionCheckingErrorType.LEFT_ARM_NO_IK)
        if not self.right_arm_ik_valid:
            error_types.append(GeneralExecutionCheckingErrorType.RIGHT_ARM_NO_IK)
        if not self.dual_arm_ik_valid:
            error_types.append(GeneralExecutionCheckingErrorType.DUAL_ARM_NO_IK)
        if not self.left_arm_joints_value_valid:
            error_types.append(GeneralExecutionCheckingErrorType.LEFT_ARM_JOINTS_VALUE_ERROR)
        if not self.right_arm_joints_value_valid:
            error_types.append(GeneralExecutionCheckingErrorType.RIGHT_ARM_JOINTS_VALUE_ERROR)
        if not self.left_arm_jacobian_valid:
            error_types.append(GeneralExecutionCheckingErrorType.LEFT_ARM_JACOBIAN_ERROR)
        if not self.right_arm_jacobian_valid:
            error_types.append(GeneralExecutionCheckingErrorType.RIGHT_ARM_JACOBIAN_ERROR)
        if not self.no_collision:
            error_types.append(GeneralExecutionCheckingErrorType.COLLISION_DETECTED)
        if not self.dual_arm_planning_valid:
            error_types.append(GeneralExecutionCheckingErrorType.DUAL_ARM_NO_PLANNING)
        if not self.left_arm_trajectory_valid:
            error_types.append(GeneralExecutionCheckingErrorType.LEFT_ARM_TRAJECTORY_ERROR)
        if not self.right_arm_trajectory_valid:
            error_types.append(GeneralExecutionCheckingErrorType.RIGHT_ARM_TRAJECTORY_ERROR)
        return error_types
        
    @property
    def overall_success(self):
        return (
            self.left_arm_within_workspace and
            self.right_arm_within_workspace and
            self.left_arm_ik_valid and
            self.right_arm_ik_valid and
            self.dual_arm_ik_valid and
            self.left_arm_joints_value_valid and
            self.right_arm_joints_value_valid and
            self.left_arm_jacobian_valid and
            self.right_arm_jacobian_valid and
            self.no_collision and
            self.dual_arm_planning_valid and
            self.left_arm_trajectory_valid and
            self.right_arm_trajectory_valid
        )

@dataclass
class GeneralDualArmExecutionCheckingContext:
    from controller.atom_controller import AtomController
    execution_result: GeneralDualArmExecutionCheckingResult = field(init=True)
    checker_params: omegaconf = field(init=True)
    controller: AtomController = field(init=True)
    pose_start_left: RigidTransform = field(default=None)
    pose_start_right: RigidTransform = field(default=None)
    pose_end_left: RigidTransform = field(default=None)
    pose_end_right: RigidTransform = field(default=None)
    cached_joints_value_left: list = field(default_factory=list)
    cached_joints_value_right: list = field(default_factory=list)
    cached_joints_value_left_start: list = field(default_factory=list)
    cached_joints_value_right_start: list = field(default_factory=list)
    cached_joints_value_left_end: list = field(default_factory=list)
    cached_joints_value_right_end: list = field(default_factory=list)
    cached_joints_value_left_list: list = field(default_factory=list)
    cached_joints_value_right_list: list = field(default_factory=list)
    cached_waypoints: list = field(default_factory=list)
    
    def reset(self):
        self.execution_result = GeneralDualArmExecutionCheckingResult()
        self.pose_start_left = None
        self.pose_start_right = None
        self.pose_end_left = None
        self.pose_end_right = None
        self.cached_joints_value_left = [] # for collision checking
        self.cached_joints_value_right = [] # for collision checking
        self.cached_joints_value_left_start = [] # for planning checking
        self.cached_joints_value_right_start = [] # for planning checking
        self.cached_joints_value_left_end = [] # for planning checking
        self.cached_joints_value_right_end = [] # for planning checking
        self.cached_joints_value_left_list = [] # for execution
        self.cached_joints_value_right_list = [] # for execution
        self.cached_waypoints = [] # for trajectory checking
    
    def cache(self):
        self.cached_execution_result = deepcopy(self.execution_result)
        self.cached_cached_joints_value_left = deepcopy(self.cached_joints_value_left)
        self.cached_cached_joints_value_right = deepcopy(self.cached_joints_value_right)
        self.cached_cached_joints_value_left_start = deepcopy(self.cached_joints_value_left_start)
        self.cached_cached_joints_value_right_start = deepcopy(self.cached_joints_value_right_start)
        self.cached_cached_joints_value_left_end = deepcopy(self.cached_joints_value_left_end)
        self.cached_cached_joints_value_right_end = deepcopy(self.cached_joints_value_right_end)
    
    def revert(self):
        self.execution_result = deepcopy(self.cached_execution_result)
        self.cached_joints_value_left = deepcopy(self.cached_cached_joints_value_left)
        self.cached_joints_value_right = deepcopy(self.cached_cached_joints_value_right)
        self.cached_joints_value_left_start = deepcopy(self.cached_cached_joints_value_left_start)
        self.cached_joints_value_right_start = deepcopy(self.cached_cached_joints_value_right_start)
        self.cached_joints_value_left_end = deepcopy(self.cached_cached_joints_value_left_end)
        self.cached_joints_value_right_end = deepcopy(self.cached_cached_joints_value_right_end)
    
    def tick(self):
        self.cached_joints_value_left_list.append(self.cached_joints_value_left_end)
        self.cached_joints_value_right_list.append(self.cached_joints_value_right_end)
        self.cached_joints_value_left_start = self.cached_joints_value_left_end
        self.cached_joints_value_right_start = self.cached_joints_value_right_end
        self.cached_joints_value_left_end = []
        self.cached_joints_value_right_end = []
        
    def finish(self):
        self.cached_joints_value_left_list.append(self.cached_joints_value_left_end)
        self.cached_joints_value_right_list.append(self.cached_joints_value_right_end)
    
    def assign_target_poses(self, pose_left=None, pose_right=None):
        self.pose_end_left = pose_left
        self.pose_end_right = pose_right
        
    def load_before_fling_joint_values(self):
        self.cached_joints_value_left_end = self.controller.init_parser.config.motion.before_fling_pose_l
        self.cached_joints_value_right_end = self.controller.init_parser.config.motion.before_fling_pose_r

@dataclass
class ActionMessage:
    action_type: ActionTypeDef = field(default=ActionTypeDef.DONE)
    object_type: ObjectTypeDef = field(default=ObjectTypeDef.NONE)
    pick_points: List[RigidTransform] = field(default=(RigidTransform(), RigidTransform()))
    place_points: List[RigidTransform] = field(default=(RigidTransform(), RigidTransform()))
    joints_value_list: List[np.ndarray] = field(default=(None, None))
    extra_params: Dict[str, Any] = field(default_factory=dict)
    extra_action_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.pick_points is None:
            self.pick_points = [None, None]
        if self.place_points is None:
            self.place_points = [None, None]

        assert len(self.pick_points) == 2
        assert len(self.place_points) == 2
        pass

    @property
    def left_pick_pt(self):
        return self.pick_points[0]

    @left_pick_pt.setter
    def left_pick_pt(self, value: RigidTransform):
        self.pick_points[0] = value

    @property
    def right_pick_pt(self):
        return self.pick_points[1]

    @right_pick_pt.setter
    def right_pick_pt(self, value: RigidTransform):
        self.pick_points[1] = value

    @property
    def left_place_pt(self):
        return self.place_points[0]

    @left_place_pt.setter
    def left_place_pt(self, value: RigidTransform):
        self.place_points[0] = value

    @property
    def right_place_pt(self):
        return self.place_points[1]

    @right_place_pt.setter
    def right_place_pt(self, value: RigidTransform):
        self.place_points[1] = value

    def to_dict(self):
        raise NotImplementedError

    def from_dict(self):
        raise NotImplementedError


def new_action_message() -> ActionMessage:
    return ActionMessage()

@dataclass
class ActionIteratorMessage:
    poses_4d: Optional[np.ndarray] = field(default=np.zeros((4, 4)), init=True)  # (x ,y, z, theta) for (pick_left, pick_right, place_left, place_right)
    poses_nocs: Optional[np.ndarray] = field(default=None, init=True)  # NOCS coordinates (pick_left, pick_right)
    grasp_idxs: Optional[Tuple[int, int]] = field(default=(0, 0), init=True)  # index of grasp points (pick_left, pick_right)
    extra_params: Optional[Dict[str, float]] = field(default_factory=dict, init=True)

@dataclass
class PredictionMessage:
    action_type: ActionTypeDef = field(default=ActionTypeDef.DONE)
    action_iterator: Iterable = field(default_factory=list, init=True)
    action_xyz_diffusion: np.ndarray = field(default=np.array([]))
    action_nocs_diffusion: np.ndarray = field(default=np.array([]))
    action_xyz_list_diffusion: List = field(default_factory=list)
    pc_xyz: np.ndarray = field(default=None)
    attmaps: dict = field(default=None)
    nocs_map: np.ndarray = field(default=np.array([]))
    grasp_point_all: np.ndarray = field(default=np.zeros([4, 3]))
    grasp_point_nocs_all: np.ndarray = field(default=np.zeros([4, 3]))
    virtual_reward_all: np.ndarray = field(default=np.zeros([1, 1, 1]))
    real_reward_all: np.ndarray = field(default=np.zeros([1, 1, 1]))
    nn_timing: float = field(default=0.)

@dataclass
class KeypointMessage:
    keypoint_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    rotated_keypoint_dict: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    original_learnt_keypoint_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    principle_axis_angle: float = field(default=0., init=True)  # the principle_axis angle (in rad) relative to reference pose
    rotation_matrix: np.ndarray = field(default=np.eye(3))
    centroid: np.ndarray = field(default=None)

@dataclass
class ObservationMessage:
    valid_virtual_pts: np.ndarray = field(default=None)
    valid_virtual_pcd: o3d.geometry.PointCloud = field(default=None)
    raw_virtual_pts: np.ndarray = field(default=None)
    raw_virtual_pcd: o3d.geometry.PointCloud = field(default=None)
    mask_img: np.ndarray = field(default=None)
    rgb_img: np.ndarray = field(default=None)
    projected_rgb_img: np.ndarray = field(default=None)
    projected_depth_img: np.ndarray = field(default=None)
    projected_mask_img: np.ndarray = field(default=None)
    particle_xyz: np.ndarray = field(default=None)
    valid_nocs_pts: np.ndarray = field(default=None)


class ExceptionMessage(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


if __name__ == '__main__':
    # Built-in examples and tests
    m = new_action_message()
    pass


@dataclass
class AnnotationConfig:
    annotator: str
    root_dir: str
    object_type: ObjectTypeDef
    K: int
    raw_log_namespace: str
    extra_filter: Dict[str, Any]
    api_url: Optional[str] = None
    exam_mode: bool = False
    annotation_type: str = "real_finetune"
    multi_pose_num: int = -1
    keypoint_num: int = -1

    def __post_init__(self) -> None:
        pass

class GeneralObjectState(str, Enum):
    DISORDERED = "disordered"
    ORGANIZED = "organized"
    UNKNOWN = "unknown"
    
    def to_int(self):
        if self == GeneralObjectState.DISORDERED:
            return 0
        elif self == GeneralObjectState.ORGANIZED:
            return 1
        elif self == GeneralObjectState.UNKNOWN:
            return 2
        else:
            raise NotImplementedError
        
    def from_int(self):
        if self == 0:
            return GeneralObjectState.DISORDERED
        elif self == 1:
            return GeneralObjectState.ORGANIZED
        elif self == 2:
            return GeneralObjectState.UNKNOWN
        else:
            raise NotImplementedError

class GarmentSmoothingStyle(str, Enum):
    DOWN = "down" # Collar facing down from top-view.
    UP = "up" # Collar facing up from top-view.
    LEFT_OR_RIGHT = "left_or_right" # Collar facing left or right from top-view.
    LEFT = "left" # Collar facing left from top-view.
    RIGHT = "right" # Collar facing right from top-view.
    UNKNOWN = "unknown"
    
    def to_int(self):
        if self == GarmentSmoothingStyle.DOWN:
            return 0
        elif self == GarmentSmoothingStyle.UP:
            return 1
        elif self == GarmentSmoothingStyle.LEFT:
            return 2
        elif self == GarmentSmoothingStyle.RIGHT:
            return 3
        elif self == GarmentSmoothingStyle.LEFT_OR_RIGHT:
            return 4
        elif self == GarmentSmoothingStyle.UNKNOWN:
            return 5
        else:
            raise NotImplementedError
    
    def from_int(self):
        if self == 0:
            return GarmentSmoothingStyle.DOWN
        elif self == 1:
            return GarmentSmoothingStyle.UP
        elif self == 2:
            return GarmentSmoothingStyle.LEFT
        elif self == 3:
            return GarmentSmoothingStyle.RIGHT
        elif self == 4:
            return GarmentSmoothingStyle.LEFT_OR_RIGHT
        elif self == 5:
            return GarmentSmoothingStyle.UNKNOWN
        else:
            raise NotImplementedError
        
class ObjectState:
    def __init__(self) -> None:
        self.general_object_state = GeneralObjectState.UNKNOWN
        self.garment_smoothing_style = GarmentSmoothingStyle.UNKNOWN
        self.garment_rotation_angle = None
        self.garment_keypoint_parallel = False

@dataclass
class AnnotationResult:
    annotator: str = field(default='nobody')
    general_object_state: str = field(default=GeneralObjectState.UNKNOWN)
    garment_smoothing_style: str = field(default=GarmentSmoothingStyle.UNKNOWN)
    garment_keypoints: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None, None, None]) # 4 keypoints for now
    action_type: ActionTypeDef = field(default=ActionTypeDef.FAIL)
    action_poses: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None, None, None])
    selected_grasp_point_indices: List[List[Optional[int]]] = field(default_factory=list)
    grasp_point_rankings: List[int] = field(default_factory=list)
    fling_gt_is_better_than_rest: Optional[bool] = field(default=None)
    multiple_action_poses: List[List[Optional[np.ndarray]]] = field(default_factory=lambda: [[None, None, None, None]])
    grasp_point_pair_sorted: List[int] = field(default_factory=list)
    grasp_point_pair_rest: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        pass

    def to_dict(self) -> dict:
        """to dict for saving

        Returns:
            dict: RESULT

        Desired Format:
        annotation:
            action_type: a # int
            action_poses: [[x, y, z, r, p, y],  # left pick point
                        [x, y, z, r, p, y],  # right pick point
                        [x, y, z, r, p, y],  # left place point (all zeros if non-exists)
                        [x, y, z, r, p, y]]  # right place point (all zeros if non-exists)
            selected_grasp_point_indices:
                - [p1_idx1, p1_idx2, p2_idx1, p2_idx2],  # p1_idx1, p1_idx2 are the indices of the first pair (P1) in the predicted grasp-point list (pose_virtual.prediction.begin). p2_idx1, p2_idx2 are the indices of the second pair (P2) in the predicted grasp-point list (pose_virtual.prediction.begin).
                - ...
            grasp_point_rankings:
                - ranking  # ranking_result [int] is the ranking result (0, 1, 2, 3) -> (>, <, =, ?)
                - ...
            multiple_action_poses: [action_poses, action_poses, ...]  # list of action_poses
            grasp_point_pair_sorted: [pair_idx1, pair_idx2, ...]  # sorted pair indices
            grasp_point_pair_rest: [pair_idx1, pair_idx2, ...]
        """
        res = {
            'annotator': self.annotator,
            'general_object_state': self.general_object_state,
            'garment_smoothing_style': self.garment_smoothing_style,
            'garment_keypoints': list(map(lambda x: x.tolist() if x is not None else [0., 0., 0.], self.garment_keypoints)),
            'action_type': self.action_type.value,
            'action_poses': list(map(lambda x: x.tolist() if x is not None else [0., 0., 0., 0., 0., 0.],
                                     self.action_poses)) if self.action_poses is not None else [],
            'selected_grasp_point_indices': self.selected_grasp_point_indices,
            'grasp_point_rankings': self.grasp_point_rankings,
            'fling_gt_is_better_than_rest': self.fling_gt_is_better_than_rest,
            'multiple_action_poses': list(map(lambda x: list(
                                            map(lambda y: y.tolist() if y is not None else [0., 0., 0., 0., 0., 0.], x)
                                            ), self.multiple_action_poses)),
            'grasp_point_pair_sorted': self.grasp_point_pair_sorted,
            'grasp_point_pair_rest': self.grasp_point_pair_rest
        }
        return res

    def from_dict(self, d: dict) -> Optional[Exception]:
        try:
            self.annotator = d['annotator'] if 'annotator' in d else 'nobody'
            self.general_object_state = d['general_object_state'] if 'general_object_state' in d else GeneralObjectState.UNKNOWN
            self.garment_smoothing_style = d['garment_smoothing_style'] if 'garment_smoothing_style' in d else GarmentSmoothingStyle.UNKNOWN
            self.garment_keypoints = [np.array(x) for x in d['garment_keypoints']] if 'garment_keypoints' in d else [None, None, None, None]
            self.action_type = ActionTypeDef(d['action_type'])
            self.action_poses = [np.array(x) for x in d['action_poses']]
            self.selected_grasp_point_indices = [list(map(lambda x: int(x), entry)) for entry in
                                                 d['selected_grasp_point_indices']]
            self.grasp_point_rankings = d['grasp_point_rankings']
            self.fling_gt_is_better_than_rest = d['fling_gt_is_better_than_rest']
            self.multiple_action_poses = [[np.array(x) for x in entry] for entry in d['multiple_action_poses']]
            self.grasp_point_pair_sorted = d['grasp_point_pair_sorted']
            self.grasp_point_pair_rest = d['grasp_point_pair_rest']
            return None
        except Exception as e:
            return e


class AnnotationContext:
    _curr_pcd: o3d.geometry.PointCloud
    _curr_rgb: np.ndarray
    _entry_name: str
    _raw_log: omegaconf.DictConfig
    vis: o3d.visualization.Visualizer
    annotation_result: AnnotationResult
    console: Console

    def __init__(self, io_module: Optional['AnnotatorIO']) -> None:
        self.annotation_result = AnnotationResult()
        self.io_module = io_module
        pass

    @property
    def entry_name(self) -> str:
        return self._entry_name

    @entry_name.setter
    def entry_name(self, value: str) -> None:
        self._entry_name = value
        self._raw_log, _ = self.io_module.get_raw_log(self._entry_name)
        self._curr_pcd, _ = self.io_module.get_pcd(self._entry_name)
        self._curr_rgb, _ = self.io_module.get_rgb(self._entry_name)

    @property
    def curr_pcd(self) -> o3d.geometry.PointCloud:
        return self._curr_pcd

    @property
    def curr_rgb(self) -> np.ndarray:
        return self._curr_rgb

    @property
    def raw_log(self) -> omegaconf.DictConfig:
        return self._raw_log


class AnnotationFlag(Enum):
    COMPLETED = 1
    UNCOMPLETED = 0
    CORRUPTED = 2
    UNKNOWN = 3
