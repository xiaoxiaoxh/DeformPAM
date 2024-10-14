import numpy as np
import open3d as o3d
import os
import cv2
from typing import Tuple

from harvesters.core import Harvester
from controller.configs.error_config import error_code

from loguru import logger

class PhoXiCamera:
    def __init__(self, dev_id='DVJ-086', use_external_camera=False, external_calibration_path=None,
                 vis=False, **kwargs):
        # PhotoneoTL_DEV_<ID>
        self.device_id = "PhotoneoTL_DEV_" + dev_id
        logger.info(f"Camera device ID: self.device_id")

        if os.getenv('PHOXI_CONTROL_PATH') is not None:
            self.cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + "/API/lib/photoneo.cti"
        else:
            logger.error(f"{error_code.camera_control_path_failed}: Can not find Phoxi Control PATH!")
            return
        logger.info(f"Camera control file path: {self.cti_file_path}")

        self.use_external_camera = use_external_camera
        self.vis = vis
        
        if self.use_external_camera:
            assert external_calibration_path is not None, \
                'External calibration path is required when using external camera!'
            assert os.path.exists(external_calibration_path), \
                f'External calibration path {external_calibration_path} does not exist!'
            external_intrinsics, external_extrinsics, external_distortion_coeff, external_camera_resolution = \
                self.load_calibration_txt(external_calibration_path)
            self.external_intrinsics = external_intrinsics
            self.external_extrinsics = external_extrinsics

            from third_party.mvcam.vcamera import vCameraSystem
            external_cam_sys = vCameraSystem()
            assert len(external_cam_sys) == 1, 'Can not find any external camera!'
            self.external_camera = external_cam_sys[0]
        
        self.start()
    
    def start(self):
        self.h = Harvester()
        self.h.add_file(self.cti_file_path, True, True)
        self.h.update()

        # Print out available devices
        for item in self.h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])

        print("=="*30)
        print(self.device_id)
        self.ia = self.h.create({'id_': self.device_id})
        self.features = self.ia.remote_device.node_map

        # print(dir(self.features))
        # print("TriggerMode BEFORE: ", self.features.PhotoneoTriggerMode.value)
        self.features.PhotoneoTriggerMode.value = "Software"
        # print("TriggerMode AFTER: ", self.features.PhotoneoTriggerMode.value)

        # Send every output structure
        self.features.SendTexture.value = True
        self.features.SendPointCloud.value = True
        self.features.SendNormalMap.value = True
        self.features.SendDepthMap.value = True
        self.features.SendConfidenceMap.value = True
        # self.features.SendEventMap.value = True         # MotionCam-3D exclusive
        self.features.SendColorCameraImage.value = True # MotionCam-3D Color exclusive
        
        if self.use_external_camera:
            self.external_camera.open()

    def stop(self):
        logger.info('Closing PhoXi camera!')
        self.ia.stop()
        self.ia.destroy()
        self.h.reset()
        
        if self.use_external_camera:
            self.external_camera.close()
        
    def capture_rgb(self) -> np.ndarray:
        assert self.use_external_camera, 'Capture RGB image is only available when using external camera!'
        rgb_img = self.external_camera.read()
        if self.vis:
            cv2.imshow('rgb_img, press any key to continue', 
                       cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return rgb_img

    def capture_rgb_and_pcd(self) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Capture RGB image and point cloud from PhoXi camera

        Returns:
            rgb: RGB image, (H, W, 3) uint8 ndarray
            pcd: Point cloud, o3d.geometry.PointCloud
        """
        pcd = o3d.geometry.PointCloud()

        self.ia.stop()
        self.ia.start()
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute() # trigger first frame
        buffer = self.ia.fetch()             # grab first frame


        payload = buffer.payload

        # Order is fixed on the selected output structure. Disabled fields are shown as empty components.
        # Individual structures can enabled/disabled by the following features:
        # SendTexture, SendPointCloud, SendNormalMap, SendDepthMap, SendConfidenceMap, SendEventMap (MotionCam-3D only), SendColorCameraImage (MotionCam-3D Color only)
        # payload.components[#]
        # [0] Texture
        # [1] TextureRGB
        # [2] PointCloud [X,Y,Z,...]
        # [3] NormalMap [X,Y,Z,...]
        # [4] DepthMap
        # [5] ConfidenceMap
        # [6] EventMap
        # [7] ColorCameraImage
        
        texture_rgb_component = payload.components[1]
        texture_rgb = None
        if texture_rgb_component.width > 0 and texture_rgb_component.height > 0:
            # Reshape 1D array to 2D array with image size
            texture_rgb = texture_rgb_component.data.reshape(texture_rgb_component.height, texture_rgb_component.width, 3).copy()
            texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            texture_rgb_screen = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR)
            # Show image
            # cv2.imshow("TextureRGB", texture_rgb_screen)
            # cv2.imwrite('TextureRGB.png', texture_rgb_screen)

        # Point Cloud
        point_cloud_component = payload.components[2]
        # Normal Map
        norm_component = payload.components[3]
        # Visualize point cloud
        if point_cloud_component.width > 0 and point_cloud_component.height > 0:
            # Reshape for Open3D visualization to N x 3 arrays
            point_cloud = point_cloud_component.data.reshape(point_cloud_component.height * point_cloud_component.width, 3).copy()
            norm_map = norm_component.data.reshape(norm_component.height * norm_component.width, 3).copy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud * 0.001)
            pcd.normals = o3d.utility.Vector3dVector(norm_map)

            if texture_rgb is not None:
                rgb = cv2.convertScaleAbs(texture_rgb_screen, alpha=(255.0/65536.0))
                color_xyz = np.zeros((point_cloud_component.height * point_cloud_component.width, 3))
                color_xyz[:, 0] = np.reshape(1/65536 * texture_rgb[:, :, 0], -1)
                color_xyz[:, 1] = np.reshape(1/65536 * texture_rgb[:, :, 1], -1)
                color_xyz[:, 2] = np.reshape(1/65536 * texture_rgb[:, :, 2], -1)
                pcd.colors = o3d.utility.Vector3dVector(color_xyz)
            else:
                rgb = self.capture_rgb()

            if self.vis:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, coord])

        return rgb, pcd

    def colorize_point_cloud(
            self,
            pc: np.ndarray,
            rgb_img: np.ndarray,
    ) -> np.ndarray:
        """
        Colorize point cloud from RGB image of an external camera

        Input:
            pc: point_cloud, (N, 3) float32 ndarray
            rgb_img: 2D RGB image, (H, W, 3) uint8 ndarray
        Returns:
            pc_rgb: colors of point cloud, (N, 3) uint8 ndarray
        """
        width = rgb_img.shape[1]
        height = rgb_img.shape[0]

        # Create transformation matrix from point cloud to external camera
        inverse_trans_mat = np.linalg.inv(self.external_extrinsics)  # (4, 4)

        # Project point cloud with external camera intrinsic matrix
        pc_hom = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32)))  # (N, 4)
        pc_camera = (inverse_trans_mat @ pc_hom.T).T  # (N, 4)
        pc_image = (self.external_intrinsics @ pc_camera[:, :3].T).T  # (N, 3)
        pc_uv = pc_image[:, :2] / pc_camera[:, 2][:, np.newaxis]  # (N, 2)

        # Colorize point cloud
        num_pts = pc.shape[0]
        pc_rgb = np.zeros((num_pts, 3), dtype=np.uint8)  # (N, 3)
        valid_idxs = (pc_uv[:, 0] >= 0) & (pc_uv[:, 0] < width) & (pc_uv[:, 1] >= 0) & (pc_uv[:, 1] < height)  # (N, )
        valid_uv = np.floor(pc_uv[valid_idxs, :]).astype(np.int32)  # (N, 2)
        pc_rgb[valid_idxs, :] = rgb_img[valid_uv[:, 1], valid_uv[:, 0], :]  # (N, 3)
        return pc_rgb
    
    @staticmethod
    def load_calibration_txt(txt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load calibration information for external RGB camrea"""
        with open(txt_path, 'r') as f:
            num_list = f.readline().split(' ')[:-1]
            intrinsics = np.array([float(num) for num in num_list], dtype=np.float32).reshape(3, 3)
            num_list = f.readline().split(' ')[:-1]
            distortion_coeff = np.array([float(num) for num in num_list], dtype=np.float32)
            num_list = f.readline().split(' ')[:-1]
            rotation_matrix = np.array([float(num) for num in num_list], dtype=np.float32).reshape(3, 3)
            num_list = f.readline().split(' ')[:-1]
            translation_vector = np.array([float(num) for num in num_list], dtype=np.float32) / 1000.0  # mm -> m
            num_list = f.readline().split(' ')
            camera_resolution = np.array([int(num) for num in num_list], dtype=np.float32)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = translation_vector
        return intrinsics, extrinsics, distortion_coeff, camera_resolution
    
    

if __name__ == '__main__':
    use_external_camera = True
    camera = PhoXiCamera(dev_id='2020-12-039-LC3',vis=True, use_external_camera=use_external_camera)
    try:
        for i in range(5):
            print(f'Tring to capture {i}-th point cloud!')
            rgb, pcd = camera.capture_rgb_and_pcd()
            if use_external_camera:
                rgb = camera.capture_rgb()
    finally:
        logger.info("Can not setup camera")