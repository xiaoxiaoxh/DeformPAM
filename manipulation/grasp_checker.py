import numpy as np
import open3d as o3d
from omegaconf import DictConfig
from typing import Tuple, Optional
from loguru import logger
from third_party.realsense.realsense_camera import RealsenseCamera

class GraspChecker:
    def __init__(self, params: DictConfig):
        self.params = params
        self.camera = RealsenseCamera(**params.camera)
        self.debug = params.debug
        
    def check_grasp(self, camera_to_world_transform: np.ndarray, pose_left: np.ndarray, pose_right: np.ndarray, robot_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the grasp is valid
        pose_left: 6D pose of the left gripper
        pose_right: 6D pose of the right gripper
        robot_type: "all_robot", "left_robot", "right_robot"

        Returns:
        - bool: True if the grasp is valid, False otherwise
        - str: error message if the grasp is invalid
        """
        color_image, pcd = self.camera.get_frames()
        pcd.transform(camera_to_world_transform)
        
        pts = np.asarray(pcd.points)
        pose_left_xyz = pose_left[:3]
        pose_right_xyz = pose_right[:3]

        box_x = self.params.box.x
        box_y = self.params.box.y
        box_z = self.params.box.z
        
        if self.debug:
            vertices_left = np.array([
                [pose_left_xyz[0] - box_x / 2, pose_left_xyz[1] + box_y / 2, pose_left_xyz[2] - box_z],
                [pose_left_xyz[0] - box_x / 2, pose_left_xyz[1] - box_y / 2, pose_left_xyz[2] - box_z],
                [pose_left_xyz[0] + box_x / 2, pose_left_xyz[1] + box_y / 2, pose_left_xyz[2] - box_z],
                [pose_left_xyz[0] + box_x / 2, pose_left_xyz[1] - box_y / 2, pose_left_xyz[2] - box_z],
                [pose_left_xyz[0] - box_x / 2, pose_left_xyz[1] + box_y / 2, pose_left_xyz[2]],
                [pose_left_xyz[0] - box_x / 2, pose_left_xyz[1] - box_y / 2, pose_left_xyz[2]],
                [pose_left_xyz[0] + box_x / 2, pose_left_xyz[1] + box_y / 2, pose_left_xyz[2]],
                [pose_left_xyz[0] + box_x / 2, pose_left_xyz[1] - box_y / 2, pose_left_xyz[2]]   
            ])
            
            vertices_right = np.array([
                [pose_right_xyz[0] - box_x / 2, pose_right_xyz[1] + box_y / 2, pose_right_xyz[2] - box_z],
                [pose_right_xyz[0] - box_x / 2, pose_right_xyz[1] - box_y / 2, pose_right_xyz[2] - box_z],
                [pose_right_xyz[0] + box_x / 2, pose_right_xyz[1] + box_y / 2, pose_right_xyz[2] - box_z],
                [pose_right_xyz[0] + box_x / 2, pose_right_xyz[1] - box_y / 2, pose_right_xyz[2] - box_z],
                [pose_right_xyz[0] - box_x / 2, pose_right_xyz[1] + box_y / 2, pose_right_xyz[2]],
                [pose_right_xyz[0] - box_x / 2, pose_right_xyz[1] - box_y / 2, pose_right_xyz[2]],
                [pose_right_xyz[0] + box_x / 2, pose_right_xyz[1] + box_y / 2, pose_right_xyz[2]],
                [pose_right_xyz[0] + box_x / 2, pose_right_xyz[1] - box_y / 2, pose_right_xyz[2]]
            ])
            
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]
            ]
                        
            box_left = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices_left),
                lines=o3d.utility.Vector2iVector(edges)
            )
            box_left.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])
            
            box_right = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices_right),
                lines=o3d.utility.Vector2iVector(edges)
            )
            box_right.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(edges))])
            
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coord, box_left, box_right])

        valid_pts_left = pts[(pts[:, 0] > pose_left_xyz[0] - box_x / 2) & (pts[:, 0] < pose_left_xyz[0] + box_x / 2) &
                             (pts[:, 1] > pose_left_xyz[1] - box_y / 2) & (pts[:, 1] < pose_left_xyz[1] + box_y / 2) &
                             (pts[:, 2] > pose_left_xyz[2] - box_z)]
        valid_pts_right = pts[(pts[:, 0] > pose_right_xyz[0] - box_x / 2) & (pts[:, 0] < pose_right_xyz[0] + box_x / 2) &
                                (pts[:, 1] > pose_right_xyz[1] - box_y / 2) & (pts[:, 1] < pose_right_xyz[1] + box_y / 2) &
                                (pts[:, 2] > pose_right_xyz[2] - box_z)]
        
        logger.debug(f"Number of valid points for left gripper: {len(valid_pts_left)}")
        logger.debug(f"Number of valid points for right gripper: {len(valid_pts_right)}")
        
        success_threshold = self.params.success_threshold

        if robot_type == "all_robot":
            if len(valid_pts_left) > success_threshold and len(valid_pts_right) > success_threshold:
                logger.info("Grasp succeeded")
                return True, None
            if len(valid_pts_left) <= success_threshold:
                logger.warning("Left gripper grasp failed")
                return False, "left_robot"
            if len(valid_pts_right) <= success_threshold:
                logger.warning("Right gripper grasp failed")
                return False, "right_robot"
        elif robot_type == "left_robot":
            if len(valid_pts_left) > success_threshold:
                return True, None
            else:
                logger.warning("Left gripper grasp failed")
                return False, "left_robot"
        elif robot_type == "right_robot":
            if len(valid_pts_right) > success_threshold:
                return True, None
            else:
                logger.warning("Right gripper grasp failed")
                return False, "right_robot"