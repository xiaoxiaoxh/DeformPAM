from atom_controller import AtomController
from omegaconf import DictConfig
from typing import Callable
from configs.error_config import error_code

class Controller(AtomController):
    def __init__(self, cfg: DictConfig, check_grasp: Callable) -> None:
        """
        You have to implement this function
        """
        super().__init__(cfg, check_grasp)
    
    def execute_dual_fling(self, poses, joint_enable=False):
        """
        You have to implement this function
        """
        return error_code.ok

    def execute_fold_one(self, poses, joint_enable=False,
                         lift_z=0.35,
                         delta_place_z=-0.25, delta_place_x=0.3,
                         delta_recover_x=0.0, delta_recover_z=-0.4):
        """
        You have to implement this function
        """
        return error_code.ok

    def execute_fold_two(self, poses, joint_enable=False, middle_high=0.2):
        """
        You have to implement this function
        """
        return error_code.ok

    def execute_drag_complaint(self, poses):
        """
        You have to implement this function
        """
        return error_code.ok
    
    def execute_lift(self, pose, robot="l", joint_enable=False):
        """
        You have to implement this function
        """
        return error_code.ok
    
    def execute_sweep(self, pose, robot_type="left_robot", joint_enable=False):
        """
        You have to implement this function
        """
        return error_code.ok

    def execute_straighten_and_place(self, poses, joint_enable=False):
        """
        You have to implement this function
        """
        return error_code.ok

    def execute_single_pick_and_place(self, pose, high, robot_type="left_robot", joint_enable=False):
        """
        You have to implement this function
        """
        return error_code.ok