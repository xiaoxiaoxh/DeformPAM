# Test script for mecheye camera
# Author: Mingshuo Han

import sys

sys.path.insert(0, "../")
import argparse

import open3d as o3d

from third_party.mech_camera.cam_mecheye import MechEyeCamera

from lib_py.base import Logging, fvr_setup  # isort: skip # noqa: F401

logger = Logging.init_app_logger()


def test_mecheye():
    parser = argparse.ArgumentParser(description="mecheye camera serial number")
    parser.add_argument(
        "--identifier",
        "-i",
        dest="identifier",
        help="mecheye camera serial number",
        default="WAA15242B4030003",
        type=str,
    )
    args = parser.parse_args()
    logger.info("Camera {} is chosen.".format(args.identifier))
    camera = MechEyeCamera(identifier=args.identifier,config="/home/flexiv/yuanzhi/5_24/DeformPAM/third_party/mech_camera/configs/camera_config.json")
    # open camera
    error = camera.open()
    if error:
        logger.error("Open mecheye camera failed. Error code: {}".format(error))
        return
    # get shape
    error, rgb_shape, depth_shape = camera.get_shape()
    logger.info("RGB shape: {}. Depth shape: {}".format(rgb_shape, depth_shape))
    error, intrinsic = camera.get_intrinsics()
    logger.info("Intrinsic: {}".format(intrinsic))


    # error, rgb_image = camera.get_rgb()
    # if error == 0 and rgb_image is not None:
    #     o3d.visualization.draw_geometries([rgb_image],window_name = "RGB image")
    # else:
    #     print(f"Failed to get RGB image with error code: {error}")

    # error,pointcloud = camera.get_data(fill_hole=True)
    # print(f"pointcloud is {pointcloud}")
    # if error == 0 and pointcloud is not None:
    #     pcd_processed = pointcloud.voxel_down_sample(voxel_size=0.02)
    #     pcd_processed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud")
    # else:
    #     print(f"Failed to get point cloud with error code: {error}")

    error,points_xyz_o3d, points_xyz_rgb_o3d = camera.capture_point_cloud_from_texture_mask()
    if error == 0 and points_xyz_o3d is not None:
        o3d.visualization.draw_geometries([points_xyz_o3d], window_name="points_xyz_o3d")
    else:
        print(f"Failed to get points_xyz_o3d with error code: {error}")

    if error == 0 and points_xyz_rgb_o3d is not None:
        o3d.visualization.draw_geometries([points_xyz_rgb_o3d], window_name="points_xyz_rgb_o3d")
    else:
        print(f"Failed to get points_xyz_rgb_o3d with error code: {error}")

    
    error = camera.close()
    if error != 0:
        print(f"Failed to close camera with error code: {error}")
    else:
        print("Close successfully...")

if __name__ == "__main__":
    test_mecheye()
