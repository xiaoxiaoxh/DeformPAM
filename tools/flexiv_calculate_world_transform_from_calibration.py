"""Calibration between dual robot, global camera and world.

Author: Danqing Lee, Han Xue, Yongyuan Wang
"""

import open3d as o3d, copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import os.path as osp
import json
import argparse
import py_cli_interaction


class CaliFlexiv():
    def __init__(self):
        pass

    def init_mesh(self):
        """Init mesh environment.
        Return: mesh
        """
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        return mesh

    def transform_to_4x4(self, pose):
        """from 6d or 7d to 4x4.
        Args:
            pose: input pose. (list) (6D or 7D or 4x4)
        Return:
            transform: pose by 4x4.
        """
        if pose.size == 16:
            return pose.reshape(4, 4)
        elif pose.size == 6:
            rotation = R.from_euler('xyz', [pose[3], pose[4], pose[5]], degrees=True).as_matrix()
            transform = np.identity(4)
            transform[0:3, 0:3] = rotation
            transform[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
            return transform
        elif pose.size == 7:
            rotation = R.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
            transform = np.identity(4)
            transform[0:3, 0:3] = rotation
            transform[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
            return transform
        else:
            raise ValueError("Pose must be 6D, 7D, or 16D")

    def Calculate_core(self, left2camera, right2camera, right2world,right2turntable):
        """Calculate transform matrix.
        Args:
            left2camera: left robot to Global camera transform.
            right2camera: right robot to Global camera transform.
            right2world:  right robot to world transform.
        Return:
            camera2world: Global camera to world transform.
            left2world: left robot to world transform.
            right2world: right robot to world transform.
            turntable2world: turntable to world transform.
        """
        # trans to 4x4
        left2camera = self.transform_to_4x4(left2camera)
        right2camera = self.transform_to_4x4(right2camera)
        camera2left = np.linalg.inv(left2camera)
        camera2right = np.linalg.inv(right2camera)

        right2world = self.transform_to_4x4(right2world)

        right2turntable = self.transform_to_4x4(right2turntable)
        turntable2right = np.linalg.inv(right2turntable)

        turntable2world = turntable2right.dot(right2world)
        world2turntable = np.linalg.inv(turntable2world)

        world2right = np.linalg.inv(right2world)

        left2camera = np.linalg.inv(camera2left)
        left2right = left2camera.dot(np.linalg.inv(right2camera))
        left2world = left2right.dot(right2world)
        world2left = np.linalg.inv(left2world)

        camera2world = np.linalg.inv(right2camera).dot(right2world)
        world2camera = np.linalg.inv(camera2world)

        # print("=============start debug===================")
        # print(f"left2world:  \n{left2world}")
        # print(f"right2world: \n{right2world}")
        # print(f"camera2world:\n{camera2world}")
        # print("=============end debug=====================")

        # moving in o3d
        mesh = self.init_mesh()
        mesh_left = copy.deepcopy(mesh).transform(world2left)
        mesh_right = copy.deepcopy(mesh).transform(world2right)
        mesh_camera = copy.deepcopy(mesh).transform(world2camera)
        mesh_turntable = copy.deepcopy(mesh).transform(world2turntable)

        o3d.visualization.draw_geometries([mesh, mesh_right, mesh_left, mesh_camera,mesh_turntable])
        return camera2world, left2world, right2world, turntable2world

    def read_calibration_json(self, file_path):
        """Read json file of calibration.
        Args:
            file_path: calibration file path.
        Return:
            list.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            return np.array(data)

    def save_calibration_json(self, file_path, data):
        """Save transformation matrix in calibration file.
        Args:
            file_path: calibration file path.
            data: transformation matrix
        """
        with open(file_path, 'w') as f:
            json.dump(data.tolist(), f)


if __name__ == "__main__":
    import os

    cal = CaliFlexiv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", type=str, default="w", choices=["v", "w"])
    parser.add_argument("--calibration_base_path", type=str, default="/mnt/nfs_shared/calibration")
    args = parser.parse_args()

    __VERSION_CANDIDATES_DIR_ = args.calibration_base_path
    __VERSION_CANDIDATES__ = list(
        filter(
            lambda x: osp.isdir(osp.join(__VERSION_CANDIDATES_DIR_, x)) and args.robot_type in x and 'latest' not in x,
            os.listdir(__VERSION_CANDIDATES_DIR_))
    )
    __VERSION_CANDIDATES__.sort()
    __VERSION__ = __VERSION_CANDIDATES__[
        py_cli_interaction.must_parse_cli_sel("select calibration version", __VERSION_CANDIDATES__)]

    # create relative soft link to the latest version
    # if osp.exists(osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest')):
    #     os.remove(osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest'))
    # os.symlink(__VERSION__, osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest'), target_is_directory=True)

    # read calibration matrix
    left_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'left_robot_to_camera_cali_7D.json')
    right_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'right_robot_to_camera_cali_7D.json')
    right_to_world_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__,
                                               'right_robot_to_world_cali_6D.json')
    right_to_turntable_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__,'right_robot_to_turntable_cali_6D.json')


    # read from json file
    left2camera = cal.read_calibration_json(left_calibration_path)
    right2camera = cal.read_calibration_json(right_calibration_path)
    right2world = cal.read_calibration_json(right_to_world_calibration_path)
    right2turntable = cal.read_calibration_json(right_to_turntable_calibration_path)

    # Defined for Robotics
    camera2world, left2world, right2world, turntable2world= cal.Calculate_core(left2camera, right2camera, right2world, right2turntable)

    world2camera = np.linalg.inv(camera2world)
    world2left = np.linalg.inv(left2world)
    world2right = np.linalg.inv(right2world)
    world2turntable = np.linalg.inv(turntable2world)

    # For DeformPAM camera
    output_dir = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__)
    cal.save_calibration_json(osp.join(output_dir, 'world_to_camera_transform.json'), world2camera)
    cal.save_calibration_json(osp.join(output_dir, 'world_to_left_robot_transform.json'), world2left)
    cal.save_calibration_json(osp.join(output_dir, 'world_to_right_robot_transform.json'), world2right)
    cal.save_calibration_json(osp.join(output_dir, 'world_to_turntable_transform.json'), world2turntable)







