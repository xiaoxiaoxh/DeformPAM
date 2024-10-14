# MechEye Camera Lib
# Author: Mingshuo Han, Yongyuan Wang

import json

import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
from MechEye import Device
from MechEye.color import Color
from pydantic import BaseModel, Field

from lib_py.base import Logging, specs_manager, typechecked
from lib_py.base.custom_types import List, Literal, Optional, Tuple
from lib_py.camera.constant import CameraStatus
from lib_py.common.fill_hole import fill_hole as smoother

from third_party.mech_camera.cam_base import CameraBase
from third_party.mech_camera.error_config import cam_code as error_code

logger = Logging.init_lib_logger("MechEyeCamera")


class CameraParameters(BaseModel):
    # MechEye camera setting parameters
    scan_2d_roi: Optional[Tuple[int, int, int, int]]  # 2d ROI in x, y, w, h.
    scan_3d_exposure: Optional[List[float]] = Field(
        maxLength=3
    )  # the exposure time(ms) of the camera to capture the 3D image.
    scan_3d_roi: Optional[Tuple[int, int, int, int]]  # ROI to capture the 3D image.
    depth_range: Optional[Tuple[int, int]]  # depth range(mm) in 3D image.
    scan_2d_exposure_mode: Optional[
        Literal["Timed", "Auto", "HDR", "Flash"]
    ]  # the camera exposure mode to capture the 2D images
    scan_2d_exposure_time: Optional[float]  # the camera exposure time(ms).
    scan_2d_hdr_exposure_sequence: Optional[
        List[float]
    ]  # the camera HDR exposure sequence(ms) in float.
    scan_2d_expected_gray_value: Optional[int]  # the expected gray value.
    scan_2d_tone_mapping_enable: Optional[
        bool
    ]  # whether gray level transformation algorithm is used or not.
    scan_2d_sharpen_factor: Optional[float]  # the image sharpen factor.
    scan_3d_gain: Optional[float]  # gain to capture the 3d image.
    fringe_contrast_threshold: Optional[
        int
    ]  # the signal contrast threshold for effective pixels.
    fringe_min_threshold: Optional[int]  # fringe_min_threshold
    cloud_outlier_removal_mode: Optional[
        Literal["Off", "Weak", "Normal", "Strong"]
    ]  # the point cloud outliers removal algorithm.
    cloud_noise_removal_mode: Optional[
        Literal["Off", "Weak", "Normal", "Strong"]
    ]  # the point cloud noise removal mode
    cloud_edge_preservation_mode: Optional[
        Literal["Sharp", "Normal", "Smooth"]
    ]  # the point cloud preservation mode
    cloud_surface_smoothing_mode: Optional[
        Literal["Off", "Normal", "Weak", "Strong"]
    ]  # the point cloud smoothing mode
    projector_fringe_coding_mode: Optional[
        Literal["Fast", "Accurate"]
    ]  # projector's fringe coding mode.
    projector_power_level: Optional[
        Literal["High", "Normal", "Low"]
    ]  # projector's powerl level.
    projector_anti_flicker_mode: Optional[
        Literal["Off", "AC50Hz", "AC60Hz"]
    ]  # projector's anti-flicker mode.


class MechEyeCamera(CameraBase):
    camera_type = "mecheye"

    @typechecked
    def __init__(
        self, identifier: str = "", config: str = "", data_type: str = "RGBDPCD"
    ):
        """MechEye Camera Class
        Args:
            identifier: string. Serial number of the device.
            config: string. .json file path relative to specs folder.
            data_type: string. Data type that camera catches
        """
        super().__init__(identifier, config, data_type)
        self.device = Device()
        self.func_dict = dir(self.device)
        self.intrinsics = {}
        self.is_opened = False
        self.config = config
        

    def open(self):
        """Open or resume the camera when switched.

        Returns:
            error: error code
        """
        if self.is_opened:
            return error_code.ok
        error = self._open()
        return error

    def start(self):
        """Start the camera stream. Not supported by MechEye camera.

        Returns:
            error code ok
        """
        logger.warning(
            "MechEye camera is always stopped when it's not being triggered."
        )
        return error_code.ok

    def stop(self):
        """Stop the camera stream. Not supported by MechEye camera.

        Returns:
            error code ok
        """
        logger.warning(
            "MechEye camera is always stopped when it's not being triggered."
        )
        return error_code.ok

    def close(self):
        """Close the camera device. Not supported by MechEye camera.

        Returns:
            error code ok
        """
        self.device.disconnect()
        self.is_opened = False
        return error_code.ok

    @typechecked
    def get_rgb(self, skip: int = 0):
        """Get rgb image.

        Args:
            skip: int. Some images are to be skipped.
        Returns:
            error: error code
            rgb: rgb image in numpy format
        """
        error = self._check_opened()
        if error:
            return error, None
        for _ in range(skip + 1):
            color_map = self.device.capture_color()
        rgb = color_map.data()
        rgb_image = o3d.geometry.Image(rgb)
        return error_code.ok, rgb_image

    @typechecked
    def get_depth(self, skip: int = 0, fill_hole: bool = False):
        """Get depth image.

        Args:
            skip: int. Some images are to be skipped.
            fill_hole: bool. Whether use hole filling for depth image.
        Returns:
            error: error code
            depth: depth image in numpy format
        """
        error = self._check_opened()
        if error:
            return error, None
        for _ in range(skip + 1):
            depth_map = self.device.capture_depth()
        depth = depth_map.data()
        if fill_hole:
            depth = smoother(depth)
        return error_code.ok, depth

    @typechecked
    def get_images(self, skip: int = 0, fill_hole: bool = False):
        """Get rgb and depth images.

        Args:
            skip: int. Some images are to be skipped.
            fill_hole: bool. Whether use hole filling for depth image.
        Returns:
            error: error code
            rgb: rgb image in numpy format or None
            depth: depth image in numpy format or None
        """
        error = self._check_opened()
        if error:
            return error, None, None
        points, rgb = self._capture_data(skip=skip)
        depth = points[:, :, 2]
        if fill_hole:
            depth = smoother(depth)
        return error_code.ok, rgb, depth

    @typechecked
    def get_data(self, skip: int = 0, fill_hole: bool = False):
        """Get rgb, depth and pointcloud data.

        Args:
            skip: int. Some images are to be skipped.
            fill_hole: bool. Whether use hole filling for depth image.
        Returns:
            error: error code
            
            pcd: open3d.geometry.PointCloud. Open3d pointcloud data.
        """
        error = self._check_opened()
        if error:
            return error, None, None, None
        points, rgb = self._capture_data(skip=skip)
        depth = points[:, :, 2]
        if fill_hole:
            depth = smoother(depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            points.reshape(-1, 3).astype(np.float64)
        )
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3).astype(np.float64) / 255.0)
        return error_code.ok, pcd

    def _capture_data(self, skip):
        for _ in range(skip + 1):
            points_xyz_bgr = self.device.capture_point_xyz_bgr()
        points_xyz_bgr_data = points_xyz_bgr.data()
        shape = points_xyz_bgr_data.shape
        # This is so weired that inside or outside docker the dtype of
        # data is different. So we left two type of data process workflow.
        # TODO: (Mingshuo) Check this issue.
        if len(shape) == 3:
            points_xyz = points_xyz_bgr_data[:, :, :3]
            color = points_xyz_bgr_data[:, :, 3:].astype(np.uint8)
        else:
            height, width = shape
            points_xyz = np.zeros((height, width, 3))
            points_xyz[:, :, 0] = points_xyz_bgr_data["x"]
            points_xyz[:, :, 1] = points_xyz_bgr_data["y"]
            points_xyz[:, :, 2] = points_xyz_bgr_data["z"]
            color = np.zeros((height, width, 3))
            color[:, :, 0] = points_xyz_bgr_data["r"]
            color[:, :, 1] = points_xyz_bgr_data["g"]
            color[:, :, 2] = points_xyz_bgr_data["b"]
            color = color.astype(np.uint8)
        return points_xyz, color

    def get_status(self):
        """Get camera's status.

        Returns:
            error: error code
            status: string. camera status
        """
        error = self._check_opened()
        if error:
            return error, CameraStatus.Close
        return error_code.ok, CameraStatus.Running

    def get_intrinsics(self):
        """Get intrinsics of the camera."""
        error = self._check_opened()
        if error:
            return error, None
        _, color_shape, _ = self.get_shape()
        device_intrinsics = self.device.get_device_intrinsic()
        depth_intrinsics = device_intrinsics.depth_camera_intrinsic()
        self.intrinsics["width"], self.intrinsics["height"] = color_shape
        self.intrinsics["fx"] = depth_intrinsics.camera_matrix_fx()
        self.intrinsics["fy"] = depth_intrinsics.camera_matrix_fy()
        self.intrinsics["ppx"] = depth_intrinsics.camera_matrix_cx()
        self.intrinsics["ppy"] = depth_intrinsics.camera_matrix_cy()
        self.intrinsics["coeffs"] = [
            depth_intrinsics.dist_coeffs_k1(),
            depth_intrinsics.dist_coeffs_k2(),
            depth_intrinsics.dist_coeffs_p1(),
            depth_intrinsics.dist_coeffs_p2(),
            depth_intrinsics.dist_coeffs_k3(),
        ]
        self.intrinsics["depth_scale"] = 1.0
        return error_code.ok, self.intrinsics

    def get_shape(self):
        """Get shape in camera setting."""
        error = self._check_opened()
        if error:
            return error, None
        res = self.device.get_device_resolution()
        rgb_width = int(res.color_width())
        rgb_height = int(res.color_height())
        depth_width = int(res.depth_width())
        depth_height = int(res.depth_height())
        return error_code.ok, (rgb_width, rgb_height), (depth_width, depth_height)

    def _open(self):
        error = self._find_devices()
        if error:
            return error
        if self.config:
            error = self._load_setting()
            if error:
                return error
        self.is_opened = True
        return error

    def _find_devices(self):
        """Find specific or random device."""
        device_list = self.device.get_device_list()
        if len(device_list) == 0:
            error = error_code.mecheye_camera_no_camera_found
            logger.error(f"Error: {error}. No MechEye camera found.")
            return error
        device_index = -1
        if len(self.identifier) != 0:
            for i, info in enumerate(device_list):
                if info.id == self.identifier:
                    device_index = i
                    break
        else:
            logger.warning("No identifier specified, using random MechEye camera.")
            device_index = 0
        if device_index == -1:
            error = error_code.mecheye_camera_no_such_device
            logger.error(f"Error: {error}. No {self.identifier} MechEye camera.")
            return error
        status = self.device.connect(device_list[device_index])
        error = self._check_error(status)
        if error:
            return error

    def _load_setting(self):
        # error, config_path = specs_manager.find_file(self.config)
        error = None
        config_path = self.config
        if error:
            return error
        with open(config_path, "r") as f:
            parameters = CameraParameters(**edict(json.load(f)))
        self._set_parameters(parameters)

    def _set_parameters(self, parameters):
        for parameter, value in parameters:
            if value is None:
                continue
            elif parameter == "scan_2d_roi":
                self._check_error(
                    self.device.set_scan_2d_roi(value[0], value[1], value[2], value[3])
                )
            elif parameter == "scan_3d_roi":
                self._check_error(
                    self.device.set_scan_3d_roi(value[0], value[1], value[2], value[3])
                )
            elif parameter == "scan_3d_exposure":
                self._check_error(self.device.set_scan_3d_exposure(value))
            elif parameter == "depth_range":
                self._check_error(self.device.set_depth_range(value[0], value[1]))
            else:
                self._set_parameter(parameter, value)

    def _set_parameter(self, parameter, value):
        for func_name in self.func_dict:
            if parameter in func_name and "set" in func_name and value is not None:
                func = getattr(self.device, func_name, None)
                if func:
                    self._check_error(func(value))
                    return
        logger.warning("Parameter {} is not supported.".format(parameter))

    def _check_error(self, status):
        """Check MechEye internal error code.

        Args:
            status: error_code in MechEye SDK.
        """
        if status.ok():
            return error_code.ok
        logger.error(
            f"Mecheye internal error!. Error Code : {status.code()}, Error Description: {status.description()}"
        )
        return error_code.mecheye_internal_error

    def _check_opened(self):
        if not self.is_opened:
            error = error_code.mecheye_camera_not_opened
            logger.error(f"Error: {error}. MechEye Camera is not opened.")
            return error
        return error_code.ok
    
    def contains(self,row,col,roi):
        return row >=roi[1] and row <= roi[1] + roi[3] and col >= roi[0] and col <= roi[0]+roi[2]

    def generate_texture_mask(self, color, roi1, roi2):
        color_data = color.data()

        for row, complex in enumerate(color_data):
            for col, RGB in enumerate(complex):
                if not self.contains(row, col, roi1) and not self.contains(row, col, roi2):
                    color_data[row, col, 0] = 0
                    color_data[row, col, 1] = 0
                    color_data[row, col, 2] = 0
                else:
                    color_data[row, col, 0] = 1
                    color_data[row, col, 1] = 1
                    color_data[row, col, 2] = 1
                

        color_mask = Color(color.from_numpy(color_data))
        return color_mask

    def capture_point_cloud_from_texture_mask(self):

        error = self._check_opened()
        if error:
            return error, None, None

        # capture frame
        color = self.device.capture_color()
        depth = self.device.capture_depth()
        device_intrinsic = self.device.get_device_intrinsic()

        # geneter texture mask
        roi1 = (color.width()/5, color.height()/5, color.width() / 2, color.height() / 2)
        roi2 = (color.width() * 2 / 5, color.height() * 2 / 5, color.width() / 2, color.height() / 2)
        color_mask = self.generate_texture_mask(color, roi1, roi2)

        #generate point cloud
        points_xyz = self.device.get_cloud_from_texture_mask(depth.impl(), color_mask.impl(), device_intrinsic.impl())
        points_xyz_data = points_xyz.data()
        points_xyz_o3d = o3d.geometry.PointCloud()
        points_xyz_o3d.points = o3d.utility.Vector3dVector(points_xyz_data.reshape(-1, 3) * 0.001)
        # o3d.visualization.draw_geometries([points_xyz_o3d])
        # o3d.io.write_point_cloud("PointCloudXYZ.ply", points_xyz_o3d)
        # print("Point cloud saved to path PointCloudXYZ.ply")

        #generate colored point cloud
        points_xyz_bgr = self.device.get_bgr_cloud_from_texture_mask(depth.impl(), color_mask.impl(), color.impl(), device_intrinsic.impl())
        points_xyz_bgr_data = points_xyz_bgr.data()
    
        points_xyz_rgb_points = points_xyz_bgr_data.reshape(-1, 6)[:, :3] * 0.001
        point_xyz_rgb_colors = points_xyz_bgr_data.reshape(-1, 6)[:, 3:6] [:, ::-1] / 255
        points_xyz_rgb_o3d = o3d.geometry.PointCloud()
        points_xyz_rgb_o3d.points = o3d.utility.Vector3dVector(points_xyz_rgb_points.astype(np.float64))
        points_xyz_rgb_o3d.colors = o3d.utility.Vector3dVector(point_xyz_rgb_colors.astype(np.float64))
        # o3d.visualization.draw_geometries([points_xyz_rgb_o3d])
        # o3d.io.write_point_cloud("PointCloudXYZRGB.ply", points_xyz_rgb_o3d)
        # print("Color point cloud saved to path PointCloudXYZRGB.ply")        

        return error, points_xyz_o3d, points_xyz_rgb_o3d