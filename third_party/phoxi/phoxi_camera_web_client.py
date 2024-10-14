import numpy as np
import open3d as o3d

import requests
import base64

from loguru import logger
import json

from PIL import Image

class PhoxiCameraWebClient:
    def __init__(self, ip, port, **kwargs):
        self.ip = ip
        self.port = port
        logger.info(f"Connected to the remote Phoxi camera at {ip}:{port}.")

    def capture_rgb_and_pcd(self):
        logger.info(f"Request to capture RGB and PCD from the remote Phoxi camera.")
        response = requests.post(f"http://{self.ip}:{self.port}/camera/capture_rgb_and_pcd")
        if response.status_code == 200:
            logger.debug(f"Received RGB and PCD data from the remote Phoxi camera.")
            response_json = json.loads(response.text)
            response_json = {k: base64.b64decode(v) for k, v in response_json.items()}

            rgb_data = response_json["rgb_data"]
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8)
            rgb_shape = response_json["rgb_shape"]
            rgb_shape = np.frombuffer(rgb_shape, dtype=np.uint32)

            rgb = rgb_data.reshape(rgb_shape)

            pcd_data = response_json["pcd_data"]
            with open("/tmp/pcd_receive.ply", "wb") as f:
                f.write(pcd_data)
            pcd = o3d.io.read_point_cloud("/tmp/pcd_receive.ply")

            logger.debug(f"RGB shape: {rgb.shape}, PCD shape: {np.asarray(pcd.points).shape}.")

            return rgb, pcd
        else:
            logger.error(f"Failed to capture RGB and PCD from the remote Phoxi camera. Detail: {json.loads(response.text)}")
            return None, None

    def close(self):
        logger.info("Close the connection to the remote Phoxi camera.")
        pass

if __name__ == "__main__":
    phoxi_camera = PhoxiCameraWebClient("192.168.2.223", 13898)
    phoxi_camera.capture_rgb_and_pcd()