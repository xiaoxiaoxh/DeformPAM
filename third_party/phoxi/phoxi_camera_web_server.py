from fastapi import FastAPI, HTTPException
import uvicorn

import numpy as np
import open3d as o3d
import base64
from loguru import logger
from third_party.phoxi.phoxi_camera import PhoXiCamera

class PhoxiCameraWebServer:
    def __init__(self, dev_id='DVJ-086'):
        self.camera = PhoXiCamera(dev_id)

        self.app = FastAPI()
        self.setup_routes()

    def __del__(self):
        self.camera.stop()

    def setup_routes(self):

        @self.app.post("/camera/capture_rgb_and_pcd")
        async def capture_rgb_and_pcd():
            return self.capture_rgb_and_pcd()

    def capture_rgb_and_pcd(self):
        if self.camera is None:
            raise HTTPException(status_code=400, detail="Camera is not open")

        try:
            rgb, pcd = self.camera.capture_rgb_and_pcd()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

        # convert rgb
        rgb_data = rgb.tobytes()
        rgb_shape = np.array(rgb.shape).astype(np.uint32).tobytes()

        # convert pcd
        o3d.io.write_point_cloud("/tmp/pcd_send.ply", pcd)
        with open("/tmp/pcd_send.ply", "rb") as f:
            pcd_data = f.read()

        response = {
            "rgb_data": rgb_data,
            "rgb_shape": rgb_shape,
            "pcd_data": pcd_data
        }

        encoded_response = {k: base64.b64encode(v).decode('utf-8') for k, v in response.items()}

        return encoded_response

camera_server = PhoxiCameraWebServer()
if __name__ == "__main__":
    uvicorn.run(camera_server.app, host="0.0.0.0", port=13898)