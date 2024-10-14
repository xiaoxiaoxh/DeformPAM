import os
import os.path as osp
import sys

abs_file_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(abs_file_dir))
sys.path.insert(0, osp.join(abs_file_dir, '..', 'lib_py'))

from loguru import logger
from typing import Tuple
from MechEye.color import Color
from MechEye.point_xyz_bgr import PointXYZBGR
import open3d as o3d
import cv2
import numpy as np

from third_party.mech_camera.cam_mecheye import MechEyeCamera


class MechMindCameraWrapper:
    def __init__(self,
                 dev_id: str,
                 config_path: str,
                 vis: bool = False,
                 **kwargs):
        assert dev_id is not None, "Please specify the device id of MechMind camera."
        assert config_path is not None, "Please specify the config path of MechMind camera."
        assert osp.exists(config_path), "Config file does not exist."
        # create camera object
        self.device = MechEyeCamera(identifier=dev_id, config=config_path)
        # start camera
        self.start()

        self.vis = vis

    def start(self):
        # open camera
        error = self.device.open()
        if error:
            logger.exception("Open mecheye camera failed. Error code: {}".format(error))
            raise Exception("Open mecheye camera failed. Error code: {}".format(error))
        logger.info("MechMind camera started.")

    def stop(self):
        self.device.close()
        logger.info("MechMind camera stopped.")

    def capture_rgb(self) -> np.ndarray:
        error, bgr_image = self.device.get_rgb()

        bgr_image = np.asarray(bgr_image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if error:
            logger.exception("Get RGB data from MechMind camera failed. Error code: {}".format(error))
            raise Exception("Get RGB data from MechMind camera failed. Error code: {}".format(error))

        if self.vis:
            cv2.imshow('rgb_img, press any key to continue', bgr_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return rgb_image

    def capture_depth(self) -> np.ndarray:
        error, depth_image = self.device.get_depth(skip=1)
        if error:
            logger.exception("Get depth data from MechMind camera failed. Error code: {}".format(error))
            raise Exception("Get depth data from MechMind camera failed. Error code: {}".format(error))
        return depth_image

    def capture_pcd(self) -> o3d.geometry.PointCloud:
        """
        Capture point cloud from MechMind camera
        """
        error, pcd = self.device.get_data(fill_hole=True)
        if error:
            logger.exception("Get point cloud data from MechMind camera failed. Error code: {}".format(error))
            raise Exception("Get point cloud data from MechMind camera failed. Error code: {}".format(error))

        # convert mm to m
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 0.001)

        if self.vis:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([coord, pcd])

        return pcd

    def capture_pcd_with_mask(self, mask: np.ndarray) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Capture point cloud from MechMind camera

        Returns:
            raw_pcd: o3d.geometry.PointCloud, raw point cloud without mask
            masked_pcd: o3d.geometry.PointCloud, point cloud with mask
        """
        def convert_to_o3d_pcd(points_xyz_bgr: PointXYZBGR) -> o3d.geometry.PointCloud:
            """
            Convert PointXYZBGR to Open3D PointCloud
            """
            points_xyz_bgr_data = points_xyz_bgr.data()
            points_xyz_rgb_points = points_xyz_bgr_data.reshape(-1, 6)[:, :3] * 0.001
            point_xyz_rgb_colors = points_xyz_bgr_data.reshape(-1, 6)[:, 3:6][:, ::-1] / 255
            points_xyz_rgb_o3d = o3d.geometry.PointCloud()
            points_xyz_rgb_o3d.points = o3d.utility.Vector3dVector(points_xyz_rgb_points.astype(np.float64))
            points_xyz_rgb_o3d.colors = o3d.utility.Vector3dVector(point_xyz_rgb_colors.astype(np.float64))
            return points_xyz_rgb_o3d

        # Capture color and depth image (this part takes about 3.5s)
        color = self.device.device.capture_color()  # Raw Color format in MechMind SDK
        depth = self.device.device.capture_depth()  # Raw Depth format in MechMind SDK
        device_intrinsic = self.device.device.get_device_intrinsic()  # raw intrinsic in MechMind SDK

        fake_mask = Color(color.from_numpy(np.ones_like(mask)))  # create fake mask
        color_mask = Color(color.from_numpy(mask))  # Convert mask to Color format in MechMind SDK

        # Get colored point cloud with mask (this part takes about 0.15s)
        masked_points_xyz_bgr: PointXYZBGR = self.device.device.get_bgr_cloud_from_texture_mask(
            depth.impl(), color_mask.impl(), color.impl(), device_intrinsic.impl())
        # Get raw colored point cloud without mask (this part takes about 0.15s)
        points_xyz_bgr: PointXYZBGR = self.device.device.get_bgr_cloud_from_texture_mask(
            depth.impl(), fake_mask.impl(), color.impl(), device_intrinsic.impl())

        raw_pcd = convert_to_o3d_pcd(points_xyz_bgr)
        masked_pcd = convert_to_o3d_pcd(masked_points_xyz_bgr)

        if self.vis:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            logger.debug(f"Visualize raw point cloud and masked point cloud.")
            o3d.visualization.draw_geometries([coord, raw_pcd])
            o3d.visualization.draw_geometries([coord, masked_pcd])

        return raw_pcd, masked_pcd


if __name__ == "__main__":
    abs_file_dir = osp.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_file_dir, 'configs', 'camera_config.json')
    camera = MechMindCameraWrapper(dev_id="WAA15242B4030003",
                                   config_path=config_path,
                                   vis=True)
    try:
        for i in range(10):
            logger.debug(f"Capture {i}th frame.")
            rgb_image = camera.capture_rgb()
            depth_image = camera.capture_depth()
            pcd = camera.capture_pcd()
    finally:
        camera.stop()


