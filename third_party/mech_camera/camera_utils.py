# Camera out-of-bbox utils functions
# Author: Chongzhao Mao

import numpy as np


class CameraUtils:
    def __init__(self):
        pass

    @staticmethod
    def camK_to_list(camK):
        if CameraUtils.camK_is_mat(camK):
            camK_list = [camK[0, 0], camK[1, 1], camK[0, -1], camK[1, -1]]
        elif CameraUtils.camK_is_list(camK):
            camK_list = camK
        else:
            raise RuntimeError("invalid camK format: %s" % str(camK))
        return camK_list

    @staticmethod
    def camK_to_mat(camK):
        if CameraUtils.camK_is_mat(camK):
            camK_mat = camK
        elif CameraUtils.camK_is_list(camK):
            camK_mat = np.array(
                [[camK[0], 0, camK[2]], [0, camK[1], camK[3]], [0, 0, 1]]
            )
        else:
            raise RuntimeError("camK: ", camK, type(camK))
        return camK_mat

    @staticmethod
    def camK_to_inv(camK):
        camK = CameraUtils.camK_to_list(camK)
        camK = np.array(
            [
                [1 / camK[0], 0, -camK[2] / camK[0]],
                [0, 1 / camK[1], -camK[3] / camK[1]],
                [0, 0, 1],
            ]
        )
        return camK

    @staticmethod
    def get_camK(camK, camK_type="mat"):
        assert camK_type in [
            "list",
            "mat",
            "inv",
        ], "[ERROR][CameraUtils] unknown camK type"
        camK = getattr(CameraUtils, "camK_to_" + camK_type)(camK)
        return camK

    @staticmethod
    def camK_is_mat(camK):
        return isinstance(camK, np.ndarray) and camK.size == 9

    @staticmethod
    def camK_is_list(camK):
        return (
            isinstance(camK, list)
            and len(camK) == 4
            and (isinstance(camK[0], float))
            or isinstance(camK[0], int)
        )

    @staticmethod
    def process_intrin(intrin):
        """camK, inv_camK = Camera.process_intrin(intrin)
        Compute camK and inv_CamK according to intrinsic object.
        Args:
            intrin: numpy array or list, camera intrinsic matrix.
        Returns:
            camK: numpy array, camera intrinsic matrix.
            inv_camK: numpy array, inverse camera intrinsic matrix.
        """
        if intrin is None:
            return None, None
        if isinstance(intrin, np.ndarray):
            fx = intrin[0, 0]
            fy = intrin[1, 1]
            ppx = intrin[0, 2]
            ppy = intrin[1, 2]
        elif isinstance(intrin, list):
            fx, fy, ppx, ppy = intrin
        else:
            raise RuntimeError(
                "[Camera]: Unsupported intrinsics type -- " + type(intrin).__name__
            )

        camK = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        inv_camK = np.array([[1 / fx, 0, -ppx / fx], [0, 1 / fy, -ppy / fy], [0, 0, 1]])
        return camK, inv_camK


# base data type map
BASE_DATA_TYPE_MAP = {
    "double": np.double,
    "float": np.float32,
    "short": np.int16,
    "uint16": np.uint16,
    "uint8": np.uint8,
}


# pcd type
class PcdType:
    # (x, y, z)
    XYZ = 3
    # (x, y, z, intensity)
    XYZI = 4
    # (x, y, z, rgb)
    XYZRGB = 6
    # (x, y, z, rgba)
    XYZRGBA = 7

    @staticmethod
    def support_type_list():
        return ["XYZ", "XYZI", "XYZRGB", "XYZRGBA"]


class PcdUtils:
    @staticmethod
    def get_pcd_type(str_pcd_type):
        try:
            result = eval("PcdType." + str_pcd_type.upper())
        except Exception:
            return None
        return result

    @staticmethod
    def get_channel(str_pcd_type):
        result = PcdUtils.get_pcd_type(str_pcd_type)
        if result is None:
            return -1

        return result
