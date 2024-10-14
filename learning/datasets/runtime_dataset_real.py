import open3d as o3d
from torch.utils.data import Subset, Dataset, DataLoader
import os
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import torch
import time
import copy
import pytorch_lightning as pl
import requests
from loguru import logger
from omegaconf import OmegaConf
from typing import Tuple, Optional
from common.datamodels import ActionTypeDef, GeneralObjectState, GarmentSmoothingStyle
from torch.utils.data import WeightedRandomSampler
from learning.datasets.weighted_sampler import SampleWeightsGenerator
import learning.datasets.augmentation_v3 as aug
import MinkowskiEngine as ME
import minio
from io import BytesIO
from copy import deepcopy

oss_client = None

from multiprocessing import Pool
from tqdm import tqdm

def calculate_corrrection_matrix(n, depth):
    """
    Calculate the correction matrix to align the table plane to the z-axis and set the depth to zero.
    """
    n = n / np.linalg.norm(n)
    theta = np.arccos(n[2])
    if np.isclose(theta, 0):
        return np.eye(3)
    
    u = np.array([n[1], -n[0], 0])
    u = u / np.linalg.norm(u)
    
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[2, 3] = - depth
    P[3, 3] = 1
    
    return P

def get_table_plane_correction_matrix(raw_pc_xyz: np.ndarray, debug=False) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_pc_xyz)

    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.05)
    plane_model, inliers = pcd_downsampled.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
    
    if debug:
        inlier_cloud = pcd_downsampled.select_by_index(inliers)
        outlier_cloud = pcd_downsampled.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud.paint_uniform_color([0, 1, 0])
        
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    [a, b, c, d] = plane_model

    normal_vector = np.array([a, b, c])
    if np.isclose(np.linalg.norm(normal_vector), 0):
        depth = 0
        normal_vector = np.array([0, 0, 1])
    else:
        depth = - d / np.linalg.norm(normal_vector)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # make sure the normal vector is pointing upwards
    if normal_vector[2] < 0:
        depth = - depth
        normal_vector = - normal_vector

    # check if the normal vector is close to z-axis
    if normal_vector[2] < 0.99:
        depth = 0
        normal_vector = np.array([0, 0, 1])
    
    correction_matrix = calculate_corrrection_matrix(normal_vector, depth)
    correction_matrix = correction_matrix.astype(np.float32)
    
    return correction_matrix

def process_sample(args):
    global oss_client
    data_sample, data_sample_path, namespace, use_oss, oss_endpoint, oss_bucket_name = args
    if oss_client is None:
        oss_client = minio.Minio(endpoint=f"{oss_endpoint}",
                                access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                                secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                secure=True)

    raw_pcd_path = osp.join(data_sample_path, data_sample[namespace]['pcd']['raw']['begin'])
    if use_oss:
        raw_pcd_path = raw_pcd_path + '.npz'
        raw_pcd_response = oss_client.get_object(oss_bucket_name, raw_pcd_path)
        raw_pc_xyz = np.load(BytesIO(raw_pcd_response.read()))['points'].astype(np.float32)
    else:
        if not osp.exists(raw_pcd_path):
            raw_pcd_path = raw_pcd_path + '.npz'
        if osp.exists(raw_pcd_path):
            raw_pc_xyz = np.load(raw_pcd_path)['points'].astype(np.float32)
        else:
            raw_pcd_path = raw_pcd_path.replace('.npz', '.ply')
            assert osp.exists(raw_pcd_path), f'raw_pcd_path: {raw_pcd_path} does not exist!'
            raw_pcd = o3d.io.read_point_cloud(raw_pcd_path)
            raw_pc_xyz = np.asarray(raw_pcd.points).astype(np.float32)

    table_plane_correction_matrix = get_table_plane_correction_matrix(raw_pc_xyz)
    return (data_sample_path, table_plane_correction_matrix)

def load_pcd(args):
    global oss_client
    data_sample, data_sample_path, namespace, use_oss, oss_endpoint, oss_bucket_name = args
    if oss_client is None:
        oss_client = minio.Minio(endpoint=f"{oss_endpoint}",
                                access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                                secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                secure=True)
        
    pcd_path = osp.join(data_sample_path, data_sample[namespace]['pcd']['processed']['begin'])
    if use_oss:
        pcd_path = pcd_path + '.npz'
        pcd_response = oss_client.get_object(oss_bucket_name, pcd_path)
        pc_xyz = np.load(BytesIO(pcd_response.read()))['points'].astype(np.float32)
    else:
        if not osp.exists(pcd_path):
            pcd_path = pcd_path + '.npz'
        if osp.exists(pcd_path):
            pc_xyz = np.load(pcd_path)['points'].astype(np.float32)
        else:
            pcd_path = pcd_path.replace('.npz', '.ply')
            assert osp.exists(pcd_path), f'pcd_path: {pcd_path} does not exist!'
            pcd = o3d.io.read_point_cloud(pcd_path)
            pc_xyz = np.asarray(pcd.points).astype(np.float32)
    
    return (data_sample_path, pc_xyz)

class RuntimeDatasetReal(Dataset):
    """RuntimeDatasetReal is a real-world dataset
    that is used to filter and generate data samples at runtime."""

    def __init__(self,
                 # logging params
                 logging_dir: str = './log',
                 namespace: str = 'experiment_real',
                 namespace_extra: Optional[str] = None, # for extra classification and detection data
                 tag: str = 'debug',
                 tag_extra: Optional[str] = None, # for extra classification and detection data
                 episode_range: tuple = (0, 20),
                 # data augmentation
                 use_augmentation: bool = True,
                 normal_aug_types: tuple = ('depth', 'affine'),
                 fling_aug_types: tuple = ('depth', 'flip', 'affine', 'random_permute'),
                 depth_scale_range: tuple = (0.8, 1.2),
                 depth_trans_range: float = (0.0, 0.0),
                 flip_lr_percent: float = 0.5,
                 flip_ud_percent: float = 0.0,
                 normal_x_trans_range: tuple = (-0.1, 0.1),
                 normal_y_trans_range: tuple = (-0.1, 0.1),
                 max_normal_rot_angle: float = 20,
                 normal_scale_range: tuple = (0.8, 1.2),
                 normal_trans_place_pose: bool = True,
                 fling_x_trans_range: tuple = (-0.2, 0.2),
                 fling_y_trans_range: tuple = (-0.15, 0.15),
                 max_fling_rot_angle: float = 30,
                 fling_scale_range: tuple = (0.8, 1.2),
                 fling_trans_place_pose: bool = False,
                 label_smoothing_value: float = 0.15,
                 use_zero_center: bool = False,
                 use_ood_points_removal: bool = False,
                 # hyper-params
                 num_pc_sample: int = 8000,
                 num_pc_sample_final: int = 4000,
                 voxel_size: float = 0.002,
                 # dataset
                 valid_primitive: tuple = ('fling',),
                 static_epoch_seed: bool = False,
                 debug: bool = False,
                 num_rankings_per_sample: int = 44,
                 num_multiple_poses: int = 10,
                 data_type: str = 'real_finetune',
                 return_multiple_poses: bool = False,
                 auto_fill_multiple_poses: bool = False,
                 use_table_plane_correction: bool = False,
                 cache_data: bool = True,
                 # database config
                 use_database: bool = True,
                 log_api: str = "/v1/logs",
                 log_endpoint: str = "http://192.168.2.223:8080",
                 # oss config
                 use_oss: bool = True,
                 oss_endpoint: str = "oss.robotflow.ai",
                 oss_bucket_name: str = "unifolding",
                 **kwargs
                 ):
        super().__init__()
        self.logging_dir = logging_dir
        self.namespace = namespace
        self.namespace_extra = namespace_extra
        self.tag = tag
        self.tag_extra = tag_extra
        self.episode_range = episode_range
        self.num_rankings_per_sample = num_rankings_per_sample
        self.num_multiple_poses = num_multiple_poses
        self.data_type = data_type
        self.return_multiple_poses = return_multiple_poses
        self.auto_fill_multiple_poses = auto_fill_multiple_poses
        self.use_table_plane_correction = use_table_plane_correction
        self.cache_data = cache_data
        self.cached_pcd = {}
        
        self.valid_primitive = tuple(valid_primitive)

        # database
        self.use_database = use_database
        self.log_api = log_api
        self.log_endpoint = log_endpoint
        
        # oss
        self.use_oss = use_oss
        self.oss_endpoint = oss_endpoint
        self.oss_bucket_name = oss_bucket_name

        # dataset
        self.static_epoch_seed = static_epoch_seed
        self.debug = debug
        
        # find all data samples
        self.data_samples_list = []
        self.data_samples_path_list = []
        self.table_plane_correction_matrix_dict = {}
        self.find_data_samples()

        # hyper-params
        self.num_pc_sample = num_pc_sample
        self.voxel_size = voxel_size
        self.num_pc_sample_final = num_pc_sample_final
        # data augmentation
        # TODO: support other primitive types
        self.label_smoothing_value = label_smoothing_value
        self.use_augmentation = use_augmentation
        self.use_ood_points_removal = use_ood_points_removal
        self.normal_aug_types = normal_aug_types
        self.fling_aug_types = fling_aug_types
        depth_scale_range = depth_scale_range
        depth_trans_range = depth_trans_range
        max_fling_rot_angle = max_fling_rot_angle
        self.transform_action_normal = None
        self.transform_action_fling = None
        if use_augmentation:
            normal_aug_list = []
            fling_aug_list = []
            if 'depth' in self.normal_aug_types:
                normal_aug_list.append(aug.DepthV3(scale_range=depth_scale_range, trans_range=depth_trans_range))
            if 'depth' in self.fling_aug_types:
                fling_aug_list.append(aug.DepthV3(scale_range=depth_scale_range, trans_range=depth_trans_range))
            assert 'flip' not in self.normal_aug_types, 'Do not support flip transforms for normal action!'
            if 'flip' in self.fling_aug_types:
                fling_aug_list.append(aug.FlipV3(lr_percent=flip_lr_percent, ud_percent=flip_ud_percent, trans_place_pose=fling_trans_place_pose))  # only flip left-right
            if 'affine' in self.normal_aug_types:
                normal_aug_list.append(aug.AffineV3(
                    x_trans_range=normal_x_trans_range,
                    y_trans_range=normal_y_trans_range,
                    rot_angle_range=(-np.pi / 180 * max_normal_rot_angle, np.pi / 180 * max_normal_rot_angle),
                    scale_range=normal_scale_range,
                    trans_place_pose=normal_trans_place_pose,
                    use_zero_center=use_zero_center,
                ))
            if 'affine' in self.fling_aug_types:
                fling_aug_list.append(aug.AffineV3(
                    x_trans_range=fling_x_trans_range,
                    y_trans_range=fling_y_trans_range,
                    rot_angle_range=(-np.pi / 180 * max_fling_rot_angle, np.pi / 180 * max_fling_rot_angle),
                    scale_range=fling_scale_range,
                    trans_place_pose=fling_trans_place_pose,
                    use_zero_center=use_zero_center,
                ))
            assert 'auto_permute' not in self.normal_aug_types, 'Do not support AutoPermutePose for normal actions!'
            if 'auto_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.AutoPermutePoseV3())
            assert 'random_permute' not in self.normal_aug_types, 'Do not support RandomPermutePose for normal actions!'
            if 'random_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.RandomPermutePoseV3())
            self.transform_action_normal = aug.SequentialV3(normal_aug_list, use_torch=True)
            self.transform_action_fling = aug.SequentialV3(fling_aug_list, use_torch=True)

        # add ratio statistics for different actions
        # ...

    def find_data_samples(self):
        """Find data samples from log files."""
        if self.use_database:  # use MongoDB database
            log_dir = osp.join(self.logging_dir, self.namespace, self.tag, 'archives')
            logger.info(f'Loading data samples from {log_dir} with database ...')
            session = requests.Session()
            url = self.log_endpoint + self.log_api
            if self.data_type == 'real_finetune' or self.data_type == 'new_pipeline_finetune':
                query_filter = {"$and": [
                    {f"metadata.{self.namespace}.tag": {"$exists": "true",  "$eq": self.tag}},
                    {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$gte": self.episode_range[0]}},
                    {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$lt": self.episode_range[1]}},
                    {f"metadata.{self.namespace}.action.type": {"$in": self.valid_primitive}},
                    {f"metadata.{self.namespace}.pcd": {"$exists": "true"}},
                    {f"metadata.{self.namespace}.annotation": {"$exists": "true"}},
                    {f"metadata.{self.namespace}.annotation.grasp_point_rankings": {"$size": self.num_rankings_per_sample}},
                    # {"annotators": {"$all": ["nobody"]}}
                ]}
            elif self.data_type == 'new_pipeline_supervised':
                query_filter = {"$and": [
                    {f"metadata.{self.namespace}.tag": {"$exists": "true",  "$eq": self.tag}},
                    {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$gte": self.episode_range[0]}},
                    {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$lt": self.episode_range[1]}},
                    {f"metadata.{self.namespace}.annotation": {"$exists": "true"}},
                    {"$or": [{"$and": [{f"metadata.{self.namespace}.action.type": {"$in": self.valid_primitive}}, 
                                      {f"metadata.{self.namespace}.annotation.multiple_action_poses": {"$size": self.num_multiple_poses}}]},
                             {f"metadata.{self.namespace}.action.type": {"$nin": self.valid_primitive}}]},
                    {f"metadata.{self.namespace}.pcd": {"$exists": "true"}},
                    # {"annotators": {"$all": ["nobody"]}}
                ]}
                if self.namespace_extra is not None and self.tag_extra is not None:
                    logger.info(f'Loading extra data samples from {log_dir} with database ...')
                    query_filter_extra = {"$and": [
                        {f"metadata.{self.namespace_extra}.tag": {"$exists": "true",  "$eq": self.tag_extra}},
                        {f"metadata.{self.namespace_extra}.episode_idx": {"$exists": "true", "$gte": self.episode_range[0]}},
                        {f"metadata.{self.namespace_extra}.episode_idx": {"$exists": "true", "$lt": self.episode_range[1]}},
                        {f"metadata.{self.namespace_extra}.annotation": {"$exists": "true"}},
                        {f"metadata.{self.namespace_extra}.pcd": {"$exists": "true"}},
                    ]}
                    query_filter = {"$or": [query_filter, query_filter_extra]}
            else:
                raise ValueError(f'Unsupported data type: {self.data_type}!')
            start_time = time.time()
            response = session.get(url, json={'identifiers': None, "extra_filter": query_filter})
            response_dict_list = [json.loads(jline) for jline in response.content.splitlines()]
            self.data_samples_list = list(map(lambda x: x["metadata"], response_dict_list))
            if self.namespace_extra is not None and self.tag_extra is not None:
                for data_sample in self.data_samples_list:
                    if self.namespace in data_sample:
                        data_sample[self.namespace_extra] = deepcopy(data_sample[self.namespace])
                    else:
                        data_sample[self.namespace] = deepcopy(data_sample[self.namespace_extra])
            if self.use_oss:
                self.data_samples_path_list = list(map(lambda x: osp.join(x["identifier"]), response_dict_list))
            else:
                self.data_samples_path_list = list(map(lambda x: osp.join(log_dir, x["identifier"]), response_dict_list))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering and loading data samples!')
            session.close()
        else:
            log_dir = osp.join(self.logging_dir, self.namespace, self.tag)
            logger.debug(f'Loading data samples from {log_dir}...')
            raw_log_files = os.listdir(log_dir)
            log_files = []
            for log_file in raw_log_files:
                annotation_path = osp.join(log_dir, log_file, 'annotation.yaml')
                if osp.exists(annotation_path):
                    log_files.append(log_file)
            log_files.sort()

            logger.debug(f'Find {len(log_files)} possible valid samples!')
            logger.debug('Loading metadata....')

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=24) as executor:
                data_generator = executor.map(lambda i:
                                              (osp.join(log_dir, log_files[i]),
                                               OmegaConf.load(osp.join(log_dir, log_files[i], 'metadata.yaml'))),
                                              range(len(log_files)))
            data_samples_all = list(data_generator)

            filter_func = lambda x: self.namespace in x[1] and 'episode_idx' in x[1][self.namespace] and 'annotation' in x[1][self.namespace] and \
                                    (self.episode_range[0] <= x[1][self.namespace].episode_idx < self.episode_range[1]) and \
                                    x[1][self.namespace].action.type != 'pick_and_place' and \
                                    len(x[1][self.namespace].annotation.grasp_point_rankings) <= 40
            self.data_samples_path_list = list(map(lambda x: x[0], filter(filter_func, data_samples_all)))
            self.data_samples_list = list(map(lambda x: x[1], filter(filter_func, data_samples_all)))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering data samples!')

        logger.info(f'Found {len(self.data_samples_list)} data samples.')

        if self.use_table_plane_correction:
            start_time = time.time()
            logger.info('Calculating table plane correction matrix...')
            args_list = [
                (data_sample, data_sample_path, self.namespace, self.use_oss, self.oss_endpoint, self.oss_bucket_name)
                for data_sample, data_sample_path in zip(self.data_samples_list, self.data_samples_path_list)
            ]

            with Pool(8) as pool:
                results = list(tqdm(pool.imap(process_sample, args_list), total=len(self.data_samples_list)))

            for data_sample_path, table_plane_correction_matrix in results:
                self.table_plane_correction_matrix_dict[data_sample_path] = table_plane_correction_matrix
            
            end_time = time.time()
            logger.info(f'Use time (s): {end_time - start_time} for calculating table plane correction matrix!')
        
        if self.cache_data:
            start_time = time.time()
            logger.info('Caching point cloud data...')

            args_list = [
                (data_sample, data_sample_path, self.namespace, self.use_oss, self.oss_endpoint, self.oss_bucket_name)
                for data_sample, data_sample_path in zip(self.data_samples_list, self.data_samples_path_list)
            ]

            with Pool(8) as pool:
                results = list(tqdm(pool.imap(load_pcd, args_list), total=len(self.data_samples_list)))
            
            for data_sample_path, pc_xyz in results:
                self.cached_pcd[data_sample_path] = pc_xyz
            
            end_time = time.time()
            logger.info(f'Use time (s): {end_time - start_time} for caching point cloud data!')

    def remove_ood_points(self, pts_xyz: np.ndarray) -> np.ndarray:
        pts_xyz_torch = torch.from_numpy(pts_xyz)
        # remove out-of-distribution points according to the mean and std
        mean = torch.mean(pts_xyz_torch, dim=0)
        std = torch.std(pts_xyz_torch, dim=0)
        in_distribution_idx = (pts_xyz_torch[:, 0] > mean[0] - 3 * std[0]) & (pts_xyz_torch[:, 0] < mean[0] + 3 * std[0]) & \
                                (pts_xyz_torch[:, 1] > mean[1] - 3 * std[1]) & (pts_xyz_torch[:, 1] < mean[1] + 3 * std[1]) & \
                                (pts_xyz_torch[:, 2] > mean[2] - 3 * std[2]) & (pts_xyz_torch[:, 2] < mean[2] + 3 * std[2])
        pts_xyz = pts_xyz_torch[in_distribution_idx].numpy()
        return pts_xyz
    
    def transform_input(self, pts_xyz: np.ndarray, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(pts_xyz.shape[0])
        # random select fixed number of points
        if all_idxs.shape[0] >= self.num_pc_sample:
            selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample, replace=False)
        else:
            np.random.seed(seed)
            np.random.shuffle(all_idxs)
            res_num = len(all_idxs) - self.num_pc_sample
            selected_idxs = np.concatenate([all_idxs, all_idxs[:res_num]], axis=0)
        pc_xyz_slim = pts_xyz[selected_idxs, :]

        # perform voxelization for Sparse ResUnet-3D
        _, sel_pc_idxs = ME.utils.sparse_quantize(pc_xyz_slim / self.voxel_size, return_index=True)
        origin_slim_pc_num = sel_pc_idxs.shape[0]
        assert origin_slim_pc_num >= self.num_pc_sample_final, f'origin_slim_pc_num: {origin_slim_pc_num} < self.num_pc_sample_final: {self.num_pc_sample_final}'
        all_idxs = np.arange(origin_slim_pc_num)
        rs = np.random.RandomState(seed=seed)
        final_selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
        sel_pc_idxs = sel_pc_idxs[final_selected_idxs]
        assert sel_pc_idxs.shape[0] == self.num_pc_sample_final
        # voxelized coords for MinkowskiEngine engine
        coords = np.floor(pc_xyz_slim[sel_pc_idxs, :] / self.voxel_size)
        feat = pc_xyz_slim[sel_pc_idxs, :]
        pts_xyz = pc_xyz_slim[sel_pc_idxs, :]
        return pts_xyz, coords, feat

    def get_poses(self, data_sample: OmegaConf):
        raw_poses = np.asarray(data_sample[self.namespace]['annotation']['action_poses'])
        pose = np.stack([raw_poses[0, :3],  # left pick
                         raw_poses[2, :3],  # left place
                         raw_poses[1, :3],  # right pick
                         raw_poses[3, :3]   # right place
                         ]).astype(np.float32)
        # TODO: use real angles
        pose_angles = np.zeros((4, 1), dtype=np.float32)
        pose = np.concatenate([pose, pose_angles], axis=-1)  # (4, 4)
        pose[np.isnan(pose)] = 0.
        return pose
    
    def get_keypoints(self, data_sample: OmegaConf):
        keypoints = np.asarray(data_sample[self.namespace]['annotation']['garment_keypoints'], dtype=np.float32)
        return keypoints

    def get_virtual_prediction_poses(self, data_sample: OmegaConf):
        grasp_point_pred = np.asarray(data_sample[self.namespace]['pose_virtual']['prediction']['begin']).astype(np.float32)  # (K, 6)
        poses = []
        for i in range(grasp_point_pred.shape[0]//2):
            pose = np.stack([grasp_point_pred[i*2, :3],  # left pick
                            grasp_point_pred[i*2, :3],  # left place (dummy)
                            grasp_point_pred[i*2+1, :3],  # right pick
                            grasp_point_pred[i*2+1, :3]   # right place (dummy)
                            ]).astype(np.float32)
            pose_angles = np.zeros((4, 1), dtype=np.float32)
            pose = np.concatenate([pose, pose_angles], axis=-1)  # (4, 4)
            pose[np.isnan(pose)] = 0.
            poses.append(pose)
        return poses
    
    def get_poses_from_multiple_poses(self, data_sample: OmegaConf, return_multiple_poses: bool = False):
        raw_poses = np.asarray(data_sample[self.namespace]['annotation']['multiple_action_poses'])
        if return_multiple_poses:
            pose = np.stack([raw_poses[:, 0, :3],  # left pick
                             raw_poses[:, 2, :3],  # left place
                             raw_poses[:, 1, :3],  # right pick
                             raw_poses[:, 3, :3]   # right place
                             ], axis=1).astype(np.float32)
            pose_angles = np.zeros((raw_poses.shape[0], 4, 1), dtype=np.float32)
            pose = np.concatenate([pose, pose_angles], axis=-1)  # (4, 4)
            pose[np.isnan(pose)] = 0.
        else:
            # randomly choose one pose
            chosen_pose_idx = np.random.randint(0, raw_poses.shape[0])
            raw_poses = raw_poses[chosen_pose_idx]
            pose = np.stack([raw_poses[0, :3],  # left pick
                            raw_poses[2, :3],  # left place
                            raw_poses[1, :3],  # right pick
                            raw_poses[3, :3]   # right place
                            ]).astype(np.float32)
            # TODO: use real angles
            pose_angles = np.zeros((4, 1), dtype=np.float32)
            pose = np.concatenate([pose, pose_angles], axis=-1)  # (4, 4)
            pose[np.isnan(pose)] = 0.
        return pose

    def get_fling_rankings(self, data_sample: OmegaConf, grasp_point_all=None):
        if grasp_point_all is None:
            grasp_point_pred = np.asarray(data_sample[self.namespace]['pose_virtual']['prediction']['begin']).astype(np.float32)  # (K, 6)
            grasp_point_all = grasp_point_pred[:, :3]  # (K, 3) we only need (x, y, z) coordinate
        # [p1_idx1, p1_idx2, p2_idx1, p2_idx2]
        selected_grasp_point_indices = \
            np.asarray(data_sample[self.namespace]['annotation']['selected_grasp_point_indices'], dtype=np.int32)  # (P, 4)
        grasp_point_pair1 = np.stack([grasp_point_all[selected_grasp_point_indices[:, 0], :],
                                      grasp_point_all[selected_grasp_point_indices[:, 1], :]],
                                     axis=0)  # (2, P, 3)
        grasp_point_pair2 = np.stack([grasp_point_all[selected_grasp_point_indices[:, 2], :],
                                      grasp_point_all[selected_grasp_point_indices[:, 3], :]],
                                     axis=0)  # (2, P, 3)
        # load rankings
        rankings = np.asarray(data_sample[self.namespace]['annotation']['grasp_point_rankings']).astype(np.int32)  # (P, )
        # 0: P1 > P2 (P1 is better)
        # 1: P1 < P2 (P2 is better)
        # 2: P1 = P2 (equally good)
        # 3: Not comparable (hard to distinguish for humans).
        num_rankings = rankings.shape[0]
        pair_scores = np.zeros((num_rankings, 2))  # (P, 2)
        pair_scores[rankings == 0, 0], pair_scores[rankings == 0, 1] = \
            1.0 - self.label_smoothing_value, self.label_smoothing_value  # P1 is better
        pair_scores[rankings == 1, 0], pair_scores[rankings == 1, 1] = \
            self.label_smoothing_value, 1.0 - self.label_smoothing_value  # P2 is better
        pair_scores[rankings == 2, :] = 0.5  # equally good
        pair_scores[rankings == 3, :] = 0.  # invalid

        return grasp_point_pair1, grasp_point_pair2, pair_scores, rankings

    def conduct_table_plane_correction(self, points: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
        original_shape = points.shape

        points = torch.from_numpy(points)
        correction_matrix = torch.from_numpy(correction_matrix)

        if original_shape[-1] == 3:
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        
        if len(points.shape) == 2:
            points[..., :3] = (points @ correction_matrix.T)[..., :3]
        elif len(points.shape) == 3:
            points[..., :3] = (points @ correction_matrix.T[None, :, :])[..., :3]
        else:
            raise ValueError(f'Unsupported points shape: {points.shape}!')
        
        if original_shape[-1] == 3:
            points = points[..., :3]

        points = points.numpy()

        return points

    def __len__(self):
        return len(self.data_samples_list)

    def __getitem__(self, index: int) -> Tuple:
        data_sample = self.data_samples_list[index]
        data_sample_path = self.data_samples_path_list[index]
        # load point cloud
        pcd_path = osp.join(data_sample_path, data_sample[self.namespace]['pcd']['processed']['begin'])
        if pcd_path in self.cached_pcd:
            pc_xyz = self.cached_pcd[pcd_path]
        else:
            pc_xyz = load_pcd((data_sample, data_sample_path, self.namespace, self.use_oss, self.oss_endpoint, self.oss_bucket_name))[1]  
        
        if self.use_table_plane_correction:
            table_plane_correction_matrix = self.table_plane_correction_matrix_dict[data_sample_path]
        else:
            table_plane_correction_matrix = np.eye(4, dtype=np.float32)
            
        pc_xyz = self.conduct_table_plane_correction(pc_xyz, table_plane_correction_matrix)
                
        if self.use_ood_points_removal:
            pts_xyz_numpy = self.remove_ood_points(pc_xyz)
        else:
            pts_xyz_numpy = pc_xyz

        action_type = ActionTypeDef(data_sample[self.namespace]['annotation']['action_type'])
        action_idx = action_type.value

        # generate action poses
        if self.data_type == 'real_finetune' or self.data_type == 'new_pipeline_finetune':
            poses_numpy = self.get_poses(data_sample)
            virtual_prediction_poses = self.get_virtual_prediction_poses(data_sample)
            poses_numpy = self.conduct_table_plane_correction(poses_numpy, table_plane_correction_matrix)
            for i in range(len(virtual_prediction_poses)):
                virtual_prediction_poses[i] = self.conduct_table_plane_correction(virtual_prediction_poses[i], table_plane_correction_matrix)
            pts_xyz_numpy_ = pts_xyz_numpy.copy()
            # transform action poses (data augmentation)
            if action_type == ActionTypeDef.FLING or action_type == ActionTypeDef.SWEEP:
                rng_state = torch.get_rng_state()
                pts_xyz_numpy, poses_numpy = self.transform_action_fling(pts_xyz_numpy_.copy(), poses_numpy)
                for i in range(len(virtual_prediction_poses)):
                    torch.set_rng_state(rng_state)
                    pts_xyz_numpy, virtual_prediction_poses[i] = self.transform_action_fling(pts_xyz_numpy_.copy(), virtual_prediction_poses[i].copy())
            else:
                rng_state = torch.get_rng_state()
                pts_xyz_numpy, poses_numpy = self.transform_action_normal(pts_xyz_numpy, poses_numpy)
                for i in range(len(virtual_prediction_poses)):
                    torch.set_rng_state(rng_state)
                    pts_xyz_numpy, virtual_prediction_poses[i] = self.transform_action_normal(pts_xyz_numpy_.copy(), virtual_prediction_poses[i].copy())
            # get fling-rankings
            virtual_prediction_grasp_point_all = np.array(virtual_prediction_poses)[:, 0:3:2, :3].reshape(-1, 3)
            grasp_point_pair1_numpy, grasp_point_pair2_numpy, grasp_pair_scores_numpy, rankings_numpy = \
                self.get_fling_rankings(data_sample, grasp_point_all=virtual_prediction_grasp_point_all)            
            # transform to torch tensor    
            grasp_point_pair1, grasp_point_pair2, grasp_pair_scores, rankings = torch.from_numpy(grasp_point_pair1_numpy), \
                torch.from_numpy(grasp_point_pair2_numpy), torch.from_numpy(grasp_pair_scores_numpy), torch.from_numpy(rankings_numpy)
        elif self.data_type == 'new_pipeline_supervised':
            if self.return_multiple_poses:
                poses_numpy = self.get_poses_from_multiple_poses(data_sample, self.return_multiple_poses)
            else:
                poses_numpy = self.get_poses(data_sample)
            poses_numpy = self.conduct_table_plane_correction(poses_numpy, table_plane_correction_matrix)
            keypoints_numpy = self.get_keypoints(data_sample)
            keypoints_num = keypoints_numpy.shape[0]
            # transform action poses and keypoints (data augmentation)
            rng_state = torch.get_rng_state()
            pts_xyz_numpy_ = pts_xyz_numpy.copy()
            if action_type == ActionTypeDef.FLING or action_type == ActionTypeDef.SWEEP:
                if self.return_multiple_poses:
                    # TODO: better implementation
                    poses_numpy_ = poses_numpy.copy()
                    for i in range(self.num_multiple_poses):
                        torch.set_rng_state(rng_state) # make sure the same random seed
                        pts_xyz_numpy, poses_numpy[i] = self.transform_action_fling(pts_xyz_numpy_.copy(), poses_numpy_[i].copy())   
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    keypoints_numpy = self.transform_action_fling(np.concatenate([pts_xyz_numpy_.copy(), keypoints_numpy.copy()], axis=0), \
                                                    poses_numpy_[0], dummy_keypoint_num=keypoints_num)[0][-keypoints_num:]
                else:
                    poses_numpy_ = poses_numpy.copy()
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    pts_xyz_numpy, poses_numpy = self.transform_action_fling(pts_xyz_numpy_.copy(), poses_numpy_)
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    keypoints_numpy = self.transform_action_fling(np.concatenate([pts_xyz_numpy_.copy(), keypoints_numpy.copy()], axis=0), \
                                                    poses_numpy_, dummy_keypoint_num=keypoints_num)[0][-keypoints_num:]
            else:
                if self.return_multiple_poses:
                    # TODO: better implementation
                    if self.auto_fill_multiple_poses:
                        if poses_numpy.shape[0] < self.num_multiple_poses:
                            poses_numpy = np.concatenate([poses_numpy] * self.num_multiple_poses, axis=0)
                    poses_numpy_ = poses_numpy.copy()
                    for i in range(self.num_multiple_poses):
                        torch.set_rng_state(rng_state) # make sure the same random seed
                        pts_xyz_numpy, poses_numpy[i] = self.transform_action_normal(pts_xyz_numpy_.copy(), poses_numpy_[i].copy())
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    keypoints_numpy = self.transform_action_normal(np.concatenate([pts_xyz_numpy_.copy(), keypoints_numpy.copy()], axis=0), \
                                                    poses_numpy_[0], dummy_keypoint_num=keypoints_num)[0][-keypoints_num:]
                else:
                    poses_numpy_ = poses_numpy.copy()
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    pts_xyz_numpy, poses_numpy = self.transform_action_normal(pts_xyz_numpy_.copy(), poses_numpy_)
                    torch.set_rng_state(rng_state) # make sure the same random seed
                    keypoints_numpy = self.transform_action_normal(np.concatenate([pts_xyz_numpy_.copy(), keypoints_numpy.copy()], axis=0), \
                                                    poses_numpy_, dummy_keypoint_num=keypoints_num)[0][-keypoints_num:]
        
        else:
            raise ValueError(f'Unsupported data type: {self.data_type}!')
                    
        # important: voxelization should be performed after data augmentation
        # sample point cloud and transform to sparse tensor
        pts_xyz_numpy, coords_numpy, feat_numpy = self.transform_input(pts_xyz_numpy, seed=index if self.static_epoch_seed else None)
        
        # transform to torch tensor
        coords = torch.from_numpy(coords_numpy)
        feat = torch.from_numpy(feat_numpy)
        pts_xyz_torch = torch.from_numpy(pts_xyz_numpy)
        poses_torch = torch.from_numpy(poses_numpy)
        gripper_points = poses_torch[..., :3]  # (4, 3) or (num_multiple_poses, 4, 3)
        # action score
        primitive_index = torch.tensor(action_idx, dtype=torch.long)
        
        general_object_state = GeneralObjectState.to_int(GeneralObjectState(data_sample[self.namespace]['annotation']['general_object_state']))
        garment_smoothing_style = GarmentSmoothingStyle.to_int(GarmentSmoothingStyle(data_sample[self.namespace]['annotation']['garment_smoothing_style']))
        
        smoothed_label = torch.tensor(general_object_state, dtype=torch.long)
        smoothing_style_label = torch.tensor(garment_smoothing_style, dtype=torch.long)
        # TODO: more flexible
        folding_score = torch.tensor(0.) if (action_type == ActionTypeDef.FOLD_1_1 or action_type == ActionTypeDef.FOLD_2) \
            else torch.tensor(1.0)

        if self.data_type == 'real_finetune':
            return coords, feat, pts_xyz_torch, gripper_points, \
                grasp_point_pair1, grasp_point_pair2, grasp_pair_scores, \
                primitive_index, smoothed_label, folding_score
        elif self.data_type == 'new_pipeline_supervised':
            # dummy data to align with the data format of imitation dataset
            pts_nocs_torch = torch.zeros_like(pts_xyz_torch)
            gripper_points_nocs = torch.zeros_like(gripper_points[:, 0:3:2]) if self.return_multiple_poses else torch.zeros_like(gripper_points[0:3:2])
            keypoints_torch = torch.from_numpy(keypoints_numpy)
            rotation_cls = torch.zeros(poses_torch.shape[:-1], dtype=torch.long)
            reward = torch.tensor(0.)
            folding_score = torch.zeros_like(folding_score)
            folding_step = torch.tensor(0)
            return coords, feat, pts_xyz_torch, pts_nocs_torch, gripper_points, gripper_points_nocs, keypoints_torch, rotation_cls, \
                reward, primitive_index, smoothed_label, smoothing_style_label, folding_score, folding_step
        elif self.data_type == 'new_pipeline_finetune':
            # dummy data to align with the data format of imitation dataset
            pts_nocs_torch = torch.zeros_like(pts_xyz_torch)
            gripper_points_nocs = torch.zeros_like(gripper_points[:, 0:3:2]) if self.return_multiple_poses else torch.zeros_like(gripper_points[0:3:2])
            rotation_cls = torch.zeros(poses_torch.shape[:-1], dtype=torch.long)
            reward = torch.tensor(0.)
            folding_score = torch.zeros_like(folding_score)
            folding_step = torch.tensor(0)
            grasp_point_nocs_pair1 = torch.zeros_like(grasp_point_pair1)
            grasp_point_nocs_pair2 = torch.zeros_like(grasp_point_pair2)
            return  coords, feat, pts_xyz_torch, pts_nocs_torch, gripper_points, gripper_points_nocs, rotation_cls, \
                reward, primitive_index, smoothed_label, smoothing_style_label, folding_score, folding_step, \
                grasp_point_pair1, grasp_point_pair2, grasp_pair_scores, grasp_point_nocs_pair1, grasp_point_nocs_pair2, rankings
        else:
            raise ValueError(f'Unsupported data type: {self.data_type}!')

class RuntimeDataModuleReal(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert (len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self, use_augmentation_in_val: bool = False):
        if self.train_dataset is not None and self.test_dataset is not None:
            return
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_dataset = RuntimeDatasetReal(**train_args)
        val_dataset = copy.copy(train_dataset)
        if not use_augmentation_in_val:
            val_dataset.use_augmentation = False
        val_dataset.static_epoch_seed = True

        data_samples_all = train_dataset.data_samples_list
        data_smaples_path_all = train_dataset.data_samples_path_list
        episode_set = set([data_sample[train_dataset.namespace]['episode_idx'] 
                           for data_sample in data_samples_all])
        
        # offset of former episodes
        offset = 0
        data_samples_all_new = []
        data_samples_path_all_new = []
        split_sample_idx_list_all = [None]*3
        for episode_idx in episode_set:
            data_samples_and_path_cur = filter(lambda x: x[0][train_dataset.namespace]['episode_idx'] == episode_idx, zip(data_samples_all, data_smaples_path_all))
            data_samples_and_path_cur = sorted(list(data_samples_and_path_cur), key=lambda x: x[0][train_dataset.namespace]["identifier"])
            data_samples_cur, data_samples_path_cur = zip(*data_samples_and_path_cur)
            data_samples_all_new.extend(data_samples_cur)
            data_samples_path_all_new.extend(data_samples_path_cur)
            
            # split for train/val/test
            num_samples = len(data_samples_cur)
            normalized_split = np.array(dataset_split)
            normalized_split = normalized_split / np.sum(normalized_split)
            sample_split = (normalized_split * num_samples).astype(np.int64)

            # add leftover instance to training set
            sample_split[0] += num_samples - np.sum(sample_split)

            # generate index for each
            all_idxs = np.arange(num_samples)
            all_idxs += offset
            offset += num_samples
            rs = np.random.RandomState(seed=split_seed)
            perm_all_idxs = rs.permutation(all_idxs)

            split_sample_idx_list_cur = list()
            prev_idx = 0
            for x in sample_split:
                next_idx = prev_idx + x
                split_sample_idx_list_cur.append(perm_all_idxs[prev_idx: next_idx])
                prev_idx = next_idx
        
            assert (np.allclose([len(x) for x in split_sample_idx_list_cur], sample_split))
            for i, split_sample_idx_list in enumerate(split_sample_idx_list_cur):
                if split_sample_idx_list_all[i] is None:
                    split_sample_idx_list_all[i] = split_sample_idx_list
                else:
                    split_sample_idx_list_all[i] = np.concatenate([split_sample_idx_list_all[i], split_sample_idx_list], axis=0)
                    
        train_dataset.data_samples_list = data_samples_all_new
        val_dataset.data_samples_list = data_samples_all_new
        train_dataset.data_samples_path_list = data_samples_path_all_new
        val_dataset.data_samples_path_list = data_samples_path_all_new

        # generate subsets
        train_idxs, val_idxs, test_idxs = split_sample_idx_list_all
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(val_dataset, test_idxs)

        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

        self.use_weighted_sampler = self.kwargs['weighted_sampler']['enable']
        sample_weights_mode = self.kwargs['weighted_sampler']['mode']
        if self.use_weighted_sampler:
            episode_idx_list_all = [data_sample[self.train_dataset.namespace]['episode_idx']
                                for data_sample in self.train_dataset.data_samples_list]
            episode_idx_list_train = [episode_idx_list_all[idx] for idx in self.train_idxs]
            weights_generator = SampleWeightsGenerator(episode_idx_list=episode_idx_list_train,
                                                       mode=sample_weights_mode,
                                                       min_weight=self.kwargs['weighted_sampler']['min_weight'])
            self.sample_weights = weights_generator.weights
        else:
            self.sample_weights = None

    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        train_subset = self.train_subset
        if self.use_weighted_sampler:
            max_sample_num = self.kwargs['weighted_sampler']['max_sample_num']
            sampler = WeightedRandomSampler(weights=self.sample_weights,
                                            num_samples=max_sample_num,
                                            replacement=True)
        else:
            sampler = None
        dataloader = DataLoader(train_subset,
                                batch_size=batch_size,
                                shuffle=not self.use_weighted_sampler,
                                num_workers=num_workers,
                                persistent_workers=False,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn,
                                sampler=sampler)
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        val_subset = self.val_subset
        dataloader = DataLoader(val_subset,
                                batch_size=min(batch_size, len(val_subset)), # avoid batch size larger than the dataset size
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn)
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        test_subset = self.test_subset
        dataloader = DataLoader(test_subset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn)
        return dataloader

    def collate_pair_fn(self, list_data: list) -> tuple:
        result_list = [[] for _ in range(len(list_data[0]) - 2)]
        coords_list = []
        feat_list = []
        for data in list_data:
            for type_idx, item in enumerate(data):
                if type_idx == 0:
                    coords_list.append(item)
                elif type_idx == 1:
                    feat_list.append(item)
                else:
                    result_list[type_idx - 2].append(item)
        final_list = [torch.stack(data) for data in result_list]
        coords, feat = ME.utils.sparse_collate(coords_list, feat_list)
        return (coords, feat) + tuple(final_list)