# Python Camera Base Library
# Author: Mingshuo Han
import cv2

from lib_py.base.custom_types import typechecked
from lib_py.base.logger import Logging

from third_party.mech_camera.camera_utils import CameraUtils
from third_party.mech_camera.error_config import cam_code as error_code

logger = Logging.init_lib_logger("CameraBase")


class CameraBase:
    camera_type = "base"

    @typechecked
    def __init__(self, identifier: str, config: str, data_type: str):
        """Camera Base Class
        Args:
            identifier: string. Serial number of the device.
            config: string. .json file path relative to specs folder.
            data_type: string. Data type that camera catch
        """
        self.camera = None
        self.config = config
        self.identifier = identifier
        self.data_type = data_type

        self.intrinsics = None

    def open(self):
        """Create camera instance, open camera device and start camera
        stream."""
        pass

    def start(self):
        """Start camera stream if the camera has relative function."""
        pass

    def stop(self):
        """Stop camera stream if the camera has relative function."""
        pass

    def close(self):
        """Close the camera device."""
        pass

    def get_rgb(self):
        """Get rgb image."""
        pass

    def get_depth(self):
        """Get depth image."""
        pass

    def get_pcd(self):
        """Get point cloud data."""
        error = error_code.not_supported
        error.set_detail(
            "Get pcd is not supported by camera type: %s, id: %s."
            % (self.camera_type, self.identifier)
        )
        logger.debug(error)
        return error_code.ok, None

    def get_intrinsics(self, camera_id):
        """Get intrinsics of the camera."""
        pass

    def get_status(self):
        """Get camera status."""
        pass

    @typechecked
    def get_data(self, skip: int = 0, fill_hole: bool = False):
        """Get rgb, depth and pcd data."""
        error, rgb, depth = self.get_images(skip, fill_hole)
        if error:
            return error, rgb, depth, None

        error, pcd = self.get_pcd()
        return error, rgb, depth, pcd

    @typechecked
    def get_images(self, skip: int = 0, fill_hole: bool = False):
        """Get rgb and depth images.

        Args:
            skip: int. Some images are to be skipped.
            fill_hole: bool. Whether use hole filling for depth image.
        Returns:
            error: error code
            rgb_np: rgb image in numpy format
            depth_np: None
        """
        error, rgb_np = self.get_rgb(skip=skip)
        if error:
            return error, None, None

        # convert rgb image channel to three
        if rgb_np.ndim == 3:
            channel = rgb_np.shape[2]
            if channel == 1:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2BGR)
            elif channel == 4:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGRA2BGR)
            elif channel == 2:
                raise NotImplementedError("Not support channel == 2.")
        elif rgb_np.ndim == 2:
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2BGR)
        else:
            raise NotImplementedError(
                "Except for 2 and 3 dimensional rgb images, "
                "rgb images of {} dimensions are not supported".format(rgb_np.ndim)
            )

        error, depth_np = self.get_depth(skip=skip)
        if error:
            return error, None, None
        return error, rgb_np, depth_np

    def get_shape(self):
        """Get shape in camera setting."""
        pass

    def get_camK(self):
        """Return camera's intrinsics in 3x3 numpy array."""
        error, intrin = self.get_intrinsics()
        if error:
            return error, None
        camK = CameraUtils.camK_to_mat(
            [intrin["fx"], intrin["fy"], intrin["ppx"], intrin["ppy"]]
        )
        return error, camK
