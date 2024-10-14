import numpy as np
import open3d as o3d


def depth_to_point(image_coord, depth, pinhole_camera_intrinsic, extrinsics):
    """Convert a point on a depth map to 3D points in world space"""
    intrinsics = pinhole_camera_intrinsic.intrinsic_matrix
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], \
                     intrinsics[0, 2], intrinsics[1, 2]
    u, v = image_coord
    u, v = float(u), float(v)
    z = float(depth)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pos_camera = np.array([x, y, z, 1.0])
    pos_world = np.linalg.inv(extrinsics) @ pos_camera
    return pos_world[:3]

def point_to_depth_map_location(point, pinhole_camera_intrinsic, extrinsics):
    """Convert a 3D point into u-v coordinate in 2D depth map"""
    intrinsics = pinhole_camera_intrinsic.intrinsic_matrix
    width = pinhole_camera_intrinsic.width
    height = pinhole_camera_intrinsic.height
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], \
        intrinsics[0, 2], intrinsics[1, 2]
    point_in_camera_coord = (extrinsics @ np.concatenate([point, np.array([1.0])])[:, np.newaxis])[:, 0]
    x, y, z = point_in_camera_coord[0], point_in_camera_coord[1], point_in_camera_coord[2]
    u = max(min(round(fx * x / z + cx), width - 1), 0)
    v = max(min(round(fy * y / z + cy), height - 1), 0)
    return u, v


def transform_point_cloud(pc: np.ndarray, matrix: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform(matrix)
    pc = np.asarray(pcd.points)
    return pc