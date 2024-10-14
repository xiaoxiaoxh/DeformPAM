import os.path as osp
import numpy as np
import open3d as o3d
import json
import os
import py_cli_interaction
from scipy.spatial.transform import Rotation as R
from hydra import initialize, compose
from hydra.utils import instantiate
import argparse

def transform_point_cloud(pc: np.ndarray, matrix: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform(matrix)
    pc = np.asarray(pcd.points)
    return pc

def get_world_transform_from_table_plane_pcd(pcd: o3d.geometry.PointCloud,
                                             reference_origin_in_camera: np.ndarray = np.array([0.0, 0.0, 0.0]),
                                             vis: bool = False) -> np.ndarray:
    print("Fitting plane to the table point cloud...")
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.02)
    plane_model, inliers = pcd_downsampled.segment_plane(distance_threshold=0.002,
                                                         ransac_n=3,
                                                         num_iterations=1000)

    if vis:
        inlier_cloud = pcd_downsampled.select_by_index(inliers)
        outlier_cloud = pcd_downsampled.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud.paint_uniform_color([0, 1, 0])
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, coord])

    [a, b, c, d] = plane_model

    normal_vector = np.array([a, b, c])
    if np.isclose(np.linalg.norm(normal_vector), 0):
        normal_vector = np.array([0, 0, 1])
    else:
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # check if the normal vector is close to z-axis
    if normal_vector[2] < 0.99:
        normal_vector = np.array([0, 0, 1])
    # make sure that the normal vector is pointing towards camera
    if normal_vector[2] > 0:
        normal_vector = -normal_vector

    z_axis_direction = normal_vector

    # select two points from the point cloud
    pick_point_indices = pick_points_from_pcd(pcd, window_name=
        "Please pick two points as Y-axis direction in world coordinate system.")
    assert len(pick_point_indices) == 2, "Please pick two points"
    pick_points = np.asarray(pcd.points)[pick_point_indices]
    y_axis_direction = pick_points[1] - pick_points[0]
    y_axis_direction = y_axis_direction / np.linalg.norm(y_axis_direction)

    # Compute the y-axis vector using the cross product of z-axis and x-axis
    x_axis_direction = np.cross(y_axis_direction, z_axis_direction)
    x_axis_direction = x_axis_direction / np.linalg.norm(x_axis_direction)

    # Form the rotation matrix
    rotation_matrix = np.column_stack((x_axis_direction, y_axis_direction, z_axis_direction))

    # calculate rotation of the world to camera transform
    ref_to_camera_transform = np.identity(4)
    ref_to_camera_transform[:3, :3] = rotation_matrix
    ref_to_camera_transform[:3, -1] = reference_origin_in_camera

    # calculate the origin in the reference frame
    table_origin_in_camera = np.asarray(pcd.points)[pick_point_indices].mean(axis=0)
    table_origin_in_reference = transform_point_cloud(table_origin_in_camera.reshape(1, 3),
                                                  ref_to_camera_transform).squeeze()

    # calculate the world to reference frame transform
    world_to_ref_transform = np.identity(4)
    world_origin_in_reference = np.zeros((3,))
    # set the z-axis of the aligned origin to be the same as the raw origin on the table
    world_origin_in_reference[2] = table_origin_in_reference[2]
    world_to_ref_transform[:3, -1] = -world_origin_in_reference

    # calculate the world to camera transform
    world_to_camera_transform = world_to_ref_transform @ ref_to_camera_transform

    return world_to_camera_transform

def read_calibration_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            print(line)
            data = line[:-1].split('\t')
            print(data)
            data = [float(item) for item in data]
            data_list.append(data)
    return np.array(data_list)

def get_pcd_from_camera(camera_version: str = "phoxi_camera_with_rgb", use_cropping: bool = True) -> o3d.geometry.PointCloud():
    assert camera_version in ('phoxi_camera_with_rgb', 'phoxi_camera_without_rgb', 'mechmind_camera')
    with initialize(config_path='../config/camera_param', version_base="1.1"):
        # config is relative to a module
        cfg = compose(config_name=f"{camera_version}")

    # create the camera
    camera = instantiate(cfg)
    # Start the camera
    if camera_version in ('phoxi_camera_with_rgb', 'phoxi_camera_without_rgb'):
        _, raw_camera_pcd = camera.capture_rgb_and_pcd()
    elif camera_version == 'mechmind_camera':
        raw_camera_pcd = camera.capture_pcd()
    else:
        raise NotImplementedError

    camera.stop()

    if use_cropping:
        # TODO: better implementation of cropping
        min_bound = [-10.0, -1.0, 1.4]
        max_bound = [10.0, 1.0, 1.7]
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        raw_camera_pcd = raw_camera_pcd.crop(aabb)
    return raw_camera_pcd


def pick_points_from_pcd(pcd: o3d.geometry.PointCloud,
                         window_name: str = "Please pick points using [shift + left click]"):
    print("")
    print(
        "1) Please pick points using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(pcd)
    vis.add_geometry(coord)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", type=str, default="v", choices=["v", "w"])
    parser.add_argument("--calibration_base_path", type=str, default="/mnt/nfs_shared/calibration")
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()
    
    __VERSION_CANDIDATES_DIR_ = args.calibration_base_path
    __VERSION_CANDIDATES__ = list(
        filter(lambda x: osp.isdir(osp.join(__VERSION_CANDIDATES_DIR_, x)) and args.robot_type in x and 'latest' not in x, 
               os.listdir(__VERSION_CANDIDATES_DIR_))
    )
    # __VERSION_CANDIDATES__.sort(key=lambda x: int(x[1:]))
    __VERSION__ = __VERSION_CANDIDATES__[py_cli_interaction.must_parse_cli_sel("select calibration version", __VERSION_CANDIDATES__)]

    if args.symlink:
        # create relative soft link to the latest version
        if osp.exists(osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest')):
            os.remove(osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest'))
        os.symlink(__VERSION__, osp.join(__VERSION_CANDIDATES_DIR_, f'{args.robot_type}_latest'), target_is_directory=True)

    # read calibration matrix
    left_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'left_robot_camera_cali_matrix.txt')
    right_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'right_robot_camera_cali_matrix.txt')
    world_to_virtual_transform_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'world_to_virtual_transform.json')
    camera_to_left_transform = read_calibration_txt(left_calibration_path)
    camera_to_right_transform = read_calibration_txt(right_calibration_path)
    with open(world_to_virtual_transform_path, 'r') as f:
        world_to_virtual_transform = np.array(json.load(f))

    camera_to_left_transform[:3, -1] = camera_to_left_transform[:3, -1] / 1000
    camera_to_right_transform[:3, -1] = camera_to_right_transform[:3, -1] / 1000
    left_to_camera_transform = np.linalg.inv(camera_to_left_transform)
    right_to_camera_transform = np.linalg.inv(camera_to_right_transform)

    # capture point cloud of the table plane
    # check whether the table has been cleaned up or not
    is_cleaned = False
    while not is_cleaned:
        is_cleaned = py_cli_interaction.parse_cli_bool("Is the table cleaned up?", default_value=True)

    # correct middle to camera transform with point cloud and plane fitting
    camera_list = ["phoxi_camera_with_rgb",
                   "phoxi_camera_without_rgb",
                   "mechmind_camera"]
    camera_type_idx = py_cli_interaction.must_parse_cli_sel("Select the camera type", camera_list)
    table_pcd_in_camera = get_pcd_from_camera(camera_version=camera_list[camera_type_idx])
    # calculate world_origin from the base of the two robots
    middle_origin = (left_to_camera_transform[:3, -1] + right_to_camera_transform[:3, -1]) / 2
    # calculate rotation matrix from the table plane
    world_to_camera_transform = get_world_transform_from_table_plane_pcd(
        table_pcd_in_camera,
        reference_origin_in_camera=middle_origin,
        vis=True)
    print(f'world_to_camera_transform: {world_to_camera_transform}')
    virtual_to_camera_transform = world_to_camera_transform @ np.linalg.inv(world_to_virtual_transform)

    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8).transform(world_to_camera_transform)
    virtual = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(virtual_to_camera_transform)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(left_to_camera_transform)
    right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(right_to_camera_transform)
    o3d.visualization.draw_geometries([table_pcd_in_camera, camera, left, right, world, virtual])  # camera coord system

    # calculate world to robot transform
    world_to_left_robot_transform = np.linalg.inv(left_to_camera_transform) @ world_to_camera_transform
    world_to_right_robot_transform = np.linalg.inv(right_to_camera_transform) @ world_to_camera_transform
    virtual_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(np.linalg.inv(world_to_virtual_transform))
    world_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    left_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(
        np.linalg.inv(world_to_left_robot_transform))
    right_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(
        np.linalg.inv(world_to_right_robot_transform))
    o3d.visualization.draw_geometries([world_in_world, left_in_world, right_in_world, virtual_in_world])  # world coord system

    left_rpy_in_world = R.from_matrix(np.linalg.inv(world_to_left_robot_transform)[:3, :3]).as_euler('xyz')
    right_rpy_in_world = R.from_matrix(np.linalg.inv(world_to_right_robot_transform)[:3, :3]).as_euler('xyz')
    print(f'left_rpy_in_world: {left_rpy_in_world}')
    print(f'right_rpy_in_world: {right_rpy_in_world}')

    output_dir = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__)
    with open(osp.join(output_dir, 'world_to_camera_transform.json'), 'w') as f:
        json.dump(world_to_camera_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_camera_transform.json')))
    with open(osp.join(output_dir, 'world_to_left_robot_transform.json'), 'w') as f:
        json.dump(world_to_left_robot_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_left_robot_transform.json')))
    with open(osp.join(output_dir, 'world_to_right_robot_transform.json'), 'w') as f:
        json.dump(world_to_right_robot_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_right_robot_transform.json')))