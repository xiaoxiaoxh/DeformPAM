import cv2
import time
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import os.path as osp
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf
import copy
import numpy as np
import open3d as o3d
import tqdm as tqdm
import json
import pymongo
from common.metric_utils import get_covered_area_from_particles, get_iou_from_2d_points, get_chamfer_distance_from_3d_points, get_earth_movers_distance_from_3d_points

import matplotlib.pyplot as plt
import seaborn as sns

import PIL
import py_cli_interaction

import multiprocessing

from typing import Optional

from sklearn.cluster import KMeans

def iou_chamfer_emd_worker(cur_pc_xyz, canonical_pc_xyz, rotation_degree):
    rotated_canonical_pc_xyz = np.dot(canonical_pc_xyz, np.array([[np.cos(np.radians(rotation_degree)), np.sin(np.radians(rotation_degree)), 0],
                                    [-np.sin(np.radians(rotation_degree)), np.cos(np.radians(rotation_degree)), 0],
                                    [0, 0, 1]]))
    iou = get_iou_from_2d_points(cur_pc_xyz[:, :2], rotated_canonical_pc_xyz[:, :2])
    chamfer_distance = get_chamfer_distance_from_3d_points(cur_pc_xyz, rotated_canonical_pc_xyz, sample_num=1000)
    emd = get_earth_movers_distance_from_3d_points(cur_pc_xyz, rotated_canonical_pc_xyz, sample_num=1000)
    return iou, chamfer_distance, emd

class LogMetricEvaluator:
    def __init__(self,
                 # logging params
                 mongodb_url,
                 mongodb_database: str = 'unifolding',
                 logging_dir: str = '/home/xuehan/DeformPAM/log',
                 namespace: str = 'experiment_new_pipeline_test',
                 tag: str = 'tshirt_short_action14_real_zero_center_new_pipeline_test',
                 namespace_canonical: str = 'capture_gt_pcd',
                 tag_canonical: str = 'test',
                 step_num_limit: int = -1,
                 trial_num_limit: Optional[int] = None,
                 recheck_status: bool = False,
                 canonical_pose_offset: tuple = (0., 0., 0.),
                 rotation_base_degrees: tuple = (0, 90, 180, 270),
                 rotation_delta_degrees: tuple = (-20, -10, 0, 10, 20),
                 debug: bool = False):
        super().__init__()
        # create MongoDB client for human evaluation database
        self.mongo_client = pymongo.MongoClient(mongodb_url)
        self.mongo_collection = self.mongo_client[mongodb_database]['logs']
        self.logging_dir = logging_dir
        self.namespace = namespace
        self.tag = tag
        self.namespace_canonical = namespace_canonical
        self.tag_canonical = tag_canonical
        assert step_num_limit > 0, 'Please specify the step_num_limit!'
        self.step_num_limit = step_num_limit
        self.trial_num_limit = trial_num_limit
        self.recheck_status = recheck_status
        self.canonical_pose_offset = np.array(canonical_pose_offset).astype(np.float32)  # offset for point cloud of canonical pose
        self.debug = debug
        self.vis_data_dir = 'vis_data'
        
        self.base_path = osp.join(self.vis_data_dir, self.tag)
        logger.add(osp.join(self.base_path, 'calculate_metrics.log'))

        self.data_samples_canonical_list = []
        self.data_samples_path_canonical_list = []
        self.processed_pcd_canonical_list = []
        self.find_canonical_data_samples()
        self.object_id_canonical_list = [data_sample[self.namespace_canonical].object_id for data_sample in self.data_samples_canonical_list]
        
        self.all_doc_info_dict_list = []
        self.filtered_doc_info_dict_list = []
        self.data_trials_list = [] # [(start, end)]
        self.data_samples_list = []
        self.data_samples_path_list = []
        self.processed_pcd_list = []
        self.find_data_samples()
        if len(self.filtered_doc_info_dict_list) == 0:
            logger.error('No valid data samples found!')
            raise ValueError('No valid data samples found!')
        
        self.success_list = []
        self.success_steps_list = []
        self.diversity_per_step_list = []
        self.normalized_coverage_per_step_list = []
        self.iou_per_step_list = []
        self.chamfer_per_step_list = []
        self.emd_per_step_list = []

        self.rotation_degrees = [base_degree + delta_degree for base_degree in rotation_base_degrees for delta_degree in rotation_delta_degrees]

    def find_canonical_data_samples(self):
        """Find data samples of canonical objects from log files."""
        log_dir = osp.join(self.logging_dir, self.namespace_canonical, self.tag_canonical)
        log_files = os.listdir(log_dir)
        log_files.sort()

        logger.debug(f'Loading data samples from {log_dir}...')
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=24) as executor:
            data_generator = executor.map(lambda i:
                                          (osp.join(log_dir, log_files[i]),
                                           OmegaConf.load(osp.join(log_dir, log_files[i], 'metadata.yaml'))),
                                          range(len(log_files)))
        data_samples_all = list(data_generator)
        filter_func = lambda x: self.namespace_canonical in x[1] and x[1][self.namespace_canonical].object_id != ''
        self.data_samples_path_canonical_list = list(map(lambda x: x[0], filter(filter_func, data_samples_all)))
        self.data_samples_canonical_list = list(map(lambda x: x[1], filter(filter_func, data_samples_all)))
        end_time = time.time()
        logger.debug(f'Use time (s): {end_time - start_time} for filtering canonical data samples!')

        logger.info(f'Found {len(self.data_samples_canonical_list)} canonical data samples.')

        for data_sample_path in self.data_samples_path_canonical_list:
            processed_pcd_path = osp.join(data_sample_path, 'pcd', 'processed_begin.npz')
            npz = np.load(processed_pcd_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(npz['points'].astype(np.float32) +
                                                    self.canonical_pose_offset[np.newaxis, :])
            pcd.colors = o3d.utility.Vector3dVector(npz['colors'].astype(np.float32) / 255.0)
            self.processed_pcd_canonical_list.append(pcd)
            
    def find_data_samples(self):
        """Find data samples of canonical objects from log files."""
        # find all data samples
        self.all_doc_info_dict_list = sorted(
            [
                {
                    'identifier': doc['identifier'],
                    'object_id': doc['metadata']['experiment_new_pipeline_test']['object_id'],
                    'timestamp': doc['metadata']['experiment_new_pipeline_test']['timestamp'],
                    'status': doc['metadata']['experiment_new_pipeline_test']['status'],
                    'action_type': doc['metadata']['experiment_new_pipeline_test']['action']['type'],
                    'prediction': doc['metadata']['experiment_new_pipeline_test']['pose_virtual']['prediction']['begin']
                }
                for doc in self.mongo_collection.find({'metadata.experiment_new_pipeline_test.tag': {"$exists": "true",  "$eq": self.tag}})
            ], 
            key = lambda x: x['timestamp']
        )
        logger.info(f'Found {len(self.all_doc_info_dict_list)} original data samples.')

        for doc_info_dict in self.all_doc_info_dict_list:
            logger.info(f"Sample identifier: {doc_info_dict['identifier']}, status: {doc_info_dict['status']}, action type: {doc_info_dict['action_type']}")

        # find valid data samples between 'begin' and 'success' or 'failed' step
        valid_idx = []
        candidate_idx = []
        start = False
        for idx, doc_info_dict in enumerate(self.all_doc_info_dict_list):
            if self.trial_num_limit is not None and len(self.data_trials_list) >= self.trial_num_limit:
                break
            if doc_info_dict['status'] == 'begin' or \
                (start == False and doc_info_dict['status'] == 'success' and doc_info_dict['action_type'] != 'done'): # success in one step 
                candidate_idx = []
                start = True
            if start and doc_info_dict['action_type'] not in ['drag', 'lift']: # skip drag and lift actions
                candidate_idx.append(idx)
                if doc_info_dict['status'] == 'success' or doc_info_dict['status'] == 'failed':
                    if len(candidate_idx) > self.step_num_limit:
                        self.all_doc_info_dict_list[idx - len(candidate_idx) + self.step_num_limit]['status'] = 'failed'
                        candidate_idx = candidate_idx[:self.step_num_limit]
                    self.data_trials_list.append((len(valid_idx), len(valid_idx) + len(candidate_idx)))
                    valid_idx.extend(candidate_idx)
                    logger.info(f"candidate idx: {candidate_idx}")
                    start = False
        self.filtered_doc_info_dict_list = [self.all_doc_info_dict_list[idx] for idx in valid_idx]
        logger.info(f'Found {len(self.filtered_doc_info_dict_list)} valid data samples.')

        # load data samples
        log_dir = osp.join(self.logging_dir, self.namespace, self.tag)
        log_files = [doc_info_dict['identifier'] for doc_info_dict in self.filtered_doc_info_dict_list]
        logger.debug(f'Loading data samples from {log_dir}...')
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=24) as executor:
            data_generator = executor.map(lambda i:
                                          (
                                            osp.join(log_dir, 'archives', log_files[i]),
                                            # OmegaConf.load(osp.join(log_dir, 'archives', log_files[i], 'metadata.yaml'))
                                          ),
                                          range(len(log_files)))
        data_samples_all = list(data_generator)
        filter_func = lambda x: True
        self.data_samples_path_list = list(map(lambda x: x[0], filter(filter_func, data_samples_all)))
        # self.data_samples_list = list(map(lambda x: x[1], filter(filter_func, data_samples_all)))
        end_time = time.time()
        logger.debug(f'Use time (s): {end_time - start_time} for loading and filtering data samples!')
        logger.info(f'Load {len(self.data_samples_path_list)} data samples.')
        
        if self.recheck_status:
            for doc_info_dict, data_sample_path in tqdm.tqdm(zip(self.filtered_doc_info_dict_list, self.data_samples_path_list), desc='Rechecking Status'):
                if doc_info_dict['status'] == 'success':
                    rgb_path = osp.join(data_sample_path, 'rgb', 'end.jpg')
                    rgb_img = PIL.Image.open(rgb_path)
                    rgb_img.show()
                    
                    __STATUS_OPTIONS__ = ['success', 'failed']
                    base = 1
                    doc_info_dict['status'] = __STATUS_OPTIONS__[py_cli_interaction.must_parse_cli_sel(
                        f"Please recheck the status of the sample {doc_info_dict['identifier']}:", 
                        __STATUS_OPTIONS__,
                        min=base
                    ) - base]
                    logger.info(f"Update status for sample {doc_info_dict['identifier']} to {doc_info_dict['status']}")

        start_time = time.time()
        for data_sample_path in tqdm.tqdm(self.data_samples_path_list, desc='Loading Point Clouds'):
            processed_pcd_path = osp.join(data_sample_path, 'pcd', 'processed_end.npz')
            npz = np.load(processed_pcd_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(npz['points'].astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(npz['colors'].astype(np.float32) / 255.0)
            self.processed_pcd_list.append(pcd)
        end_time = time.time()
        logger.debug(f'Use time (s): {end_time - start_time} for loading point clouds!')
        logger.info(f'Load {len(self.processed_pcd_list)} point clouds.')

        all_last_pcs = []
        for trial_range in self.data_trials_list:
            idx = trial_range[1] - 1
            last_pcd = self.processed_pcd_list[idx]
            last_pc = np.asarray(last_pcd.points).astype(np.float32)
            all_last_pcs.append(last_pc)
        all_last_pcs = np.array(all_last_pcs)
        np.save(osp.join(self.base_path, 'all_last_pc.npy'), all_last_pcs)
        logger.info(f'Save all last point clouds to {osp.join(self.base_path, "all_last_pc.npy")}')
        for object_id, canonical_pcd in zip(self.object_id_canonical_list, self.processed_pcd_canonical_list):
            canonical_pc = np.asarray(canonical_pcd.points).astype(np.float32)
            np.save(osp.join(self.base_path, f'{object_id}_canonical.npy'), canonical_pc)
            logger.info(f'Save canonical point cloud of object {object_id} to {osp.join(self.base_path, f"{object_id}_canonical.npy")}')

    def calc_mean_metrics(self, calc_per_object: bool = False):
        logger.info(f'Calculating Metrics for ...')
        for trial_range in tqdm.tqdm(self.data_trials_list, desc='Calculating Metrics'):
            diversity_per_step = []
            normalized_coverage_per_step = []
            iou_per_step = []
            chamfer_per_step = []
            emd_per_step = []
            for step_idx in range(*trial_range):
                pcd = self.processed_pcd_list[step_idx]
                object_id = self.filtered_doc_info_dict_list[step_idx]['object_id']
                canonical_idx = self.object_id_canonical_list.index(object_id)
                canonical_pcd = self.processed_pcd_canonical_list[canonical_idx]
                if self.debug:
                    o3d.visualization.draw_geometries([pcd, canonical_pcd])
                
                cur_pc_xyz = np.asarray(pcd.points).astype(np.float32)
                canonical_pc_xyz = np.asarray(canonical_pcd.points).astype(np.float32)
                try:
                    # calculate diversity
                    cur_prediction_xy = np.array(self.filtered_doc_info_dict_list[step_idx]['prediction'])[:, :2]
                    kmeans = KMeans(n_clusters=2, random_state=0)
                    kmeans.fit(cur_prediction_xy)
                    labels = kmeans.labels_
                    centers = kmeans.cluster_centers_
                    stds = []
                    for i in range(kmeans.n_clusters):
                        cluster_points = cur_prediction_xy[labels == i]
                        center = centers[i]
                        distances = np.linalg.norm(cluster_points - center, axis=1)
                        std = np.std(distances)
                        stds.append(std)
                    if self.debug:
                        logger.debug(f'stds: {stds}, diversity: {np.mean(stds)}')

                    # calculate coverage
                    max_coverage = get_covered_area_from_particles(canonical_pc_xyz, cloth_particle_radius=0.002)
                    cur_coverage = get_covered_area_from_particles(cur_pc_xyz, cloth_particle_radius=0.002)
                    normalized_coverage = cur_coverage / max_coverage
                    if self.debug:
                        logger.debug(f'current coverage: {cur_coverage}, max coverage: {max_coverage}, normalized coverage: {normalized_coverage}')
                    # calculate IoU and chamfer distance with considering rotation
                    with multiprocessing.Pool(processes=32) as pool:
                        results = pool.starmap(iou_chamfer_emd_worker, [(cur_pc_xyz, canonical_pc_xyz, rotation_degree) for rotation_degree in self.rotation_degrees])
                    iou = np.max([result[0] for result in results])
                    chamfer_distance = np.min([result[1] for result in results])
                    emd = np.min([result[2] for result in results])
                    
                    if self.debug:
                        logger.debug(f'IoU: {iou}')
                        logger.debug(f'chamfer distance: {chamfer_distance}')
                        logger.debug(f'EMD: {emd}')

                    diversity_per_step.append(np.mean(stds))
                    normalized_coverage_per_step.append(normalized_coverage)
                    iou_per_step.append(iou)
                    chamfer_per_step.append(chamfer_distance)
                    emd_per_step.append(emd)
                except Exception as e:
                    logger.error(e)
            self.success_list.append(self.filtered_doc_info_dict_list[trial_range[1] - 1]['status'] == 'success')
            if self.success_list[-1]:
                self.success_steps_list.append(trial_range[1] - trial_range[0])
            else:
                self.success_steps_list.append(self.step_num_limit)
            self.diversity_per_step_list.append(diversity_per_step)
            self.normalized_coverage_per_step_list.append(normalized_coverage_per_step)
            self.iou_per_step_list.append(iou_per_step)
            self.chamfer_per_step_list.append(chamfer_per_step)
            self.emd_per_step_list.append(emd_per_step)

        success_idx = [idx for idx, success in enumerate(self.success_list) if success]
        success_rate = np.mean(self.success_list)

        # 1. Calculate the mean value
        all_diversity = []
        for diversity in self.diversity_per_step_list:
            all_diversity.extend(diversity)
        mean_diversity = np.mean(all_diversity)
        mean_success_steps = np.mean([step for step in self.success_steps_list if step > 0])
        mean_iou = np.mean([self.iou_per_step_list[i][-1] for i in success_idx])
        mean_coverage = np.mean([self.normalized_coverage_per_step_list[i][-1] for i in success_idx])
        mean_chamfer = np.mean([self.chamfer_per_step_list[i][-1] for i in success_idx])
        mean_emd = np.mean([self.emd_per_step_list[i][-1] for i in success_idx])

        logger.info(f'Mean diversity: {mean_diversity}')
        logger.info(f'Mean success rate: {success_rate}, success steps: {mean_success_steps}, '
                    f'iou: {mean_iou}, normalized coverage: {mean_coverage}, chamfer distance: {mean_chamfer}, emd: {mean_emd}')

        # 2. Calculate the variance
        std_diversity = np.std(all_diversity)
        std_success_steps = np.std([step for step in self.success_steps_list if step > 0])
        std_iou = np.std([self.iou_per_step_list[i][-1] for i in success_idx])
        std_coverage = np.std([self.normalized_coverage_per_step_list[i][-1] for i in success_idx])
        std_chamfer = np.std([self.chamfer_per_step_list[i][-1] for i in success_idx])
        std_emd = np.std([self.emd_per_step_list[i][-1] for i in success_idx])

        logger.info(f'Std of diversity: {std_diversity}')
        logger.info(f'Std of success steps: {std_success_steps},'
                    f'iou: {std_iou}, normalized coverage: {std_coverage}, chamfer distance: {std_chamfer}, emd: {std_emd}')

        if calc_per_object:
            all_object_id_list = sorted(list(set([doc['object_id'] for doc in self.filtered_doc_info_dict_list])))
            success_rate_per_object = []
            success_steps_per_object = []
            iou_per_object = []
            coverage_per_object = []
            chamfer_per_object = []
            emd_per_object = []
            for object_id in all_object_id_list:
                object_idx = [idx for idx, trial in enumerate(self.data_trials_list) if self.filtered_doc_info_dict_list[trial[0]]['object_id'] == object_id]
                object_success_idx = [idx for idx in object_idx if self.success_list[idx]]
                object_success_rate = np.mean([self.success_list[idx] for idx in object_idx])
                success_rate_per_object.append(object_success_rate)

                object_diversity = []
                for idx in object_idx:
                    object_diversity.extend(self.diversity_per_step_list[idx])
                object_mean_diversity = np.mean(object_diversity)
                object_mean_success_steps = np.mean([step for step in [self.success_steps_list[idx] for idx in object_idx] if step > 0])
                object_mean_iou = np.mean([iou_per_step[-1] for iou_per_step in [self.iou_per_step_list[idx] for idx in object_success_idx]])
                object_mean_coverage = np.mean([normalized_coverage_per_step[-1] for normalized_coverage_per_step in [self.normalized_coverage_per_step_list[idx] for idx in object_success_idx]])
                object_mean_chamfer = np.mean([chamfer_per_step[-1] for chamfer_per_step in [self.chamfer_per_step_list[idx] for idx in object_success_idx]])
                object_mean_emd = np.mean([emd_per_step[-1] for emd_per_step in [self.emd_per_step_list[idx] for idx in object_success_idx]])
                success_steps_per_object.append(object_mean_success_steps)
                iou_per_object.append(object_mean_iou)
                coverage_per_object.append(object_mean_coverage)
                chamfer_per_object.append(object_mean_chamfer)
                emd_per_object.append(object_mean_emd)
                logger.info(
                    f'Object id: {object_id}, mean diversity: {object_mean_diversity}'
                )
                logger.info(
                    f'Object id: {object_id}, mean success rates: {object_success_rate}, success steps: {object_mean_success_steps}, '
                    f'iou: {object_mean_iou}, normalized coverage: {object_mean_coverage}, chamfer distance: {object_mean_chamfer}, emd: {object_mean_emd}')

                object_std_diversity = np.std(object_diversity)
                object_std_success_steps = np.std([step for step in [self.success_steps_list[idx] for idx in object_idx] if step > 0])
                object_std_iou = np.std([iou_per_step[-1] for iou_per_step in [self.iou_per_step_list[idx] for idx in object_success_idx]])
                object_std_coverage = np.std([normalized_coverage_per_step[-1] for normalized_coverage_per_step in [self.normalized_coverage_per_step_list[idx] for idx in object_success_idx]])
                object_std_chamfer = np.std([chamfer_per_step[-1] for chamfer_per_step in [self.chamfer_per_step_list[idx] for idx in object_success_idx]])
                object_std_emd = np.std([emd_per_step[-1] for emd_per_step in [self.emd_per_step_list[idx] for idx in object_success_idx]])

                logger.info(
                    f'Obejct id: {object_id}, std of diversity: {object_std_diversity}')
                logger.info(
                    f'Object id: {object_id}, std of success steps: {object_std_success_steps}, '
                    f'iou: {object_std_iou}, normalized coverage: {object_std_coverage}, chamfer distance: {object_std_chamfer}, emd: {object_std_emd}')

            std_success_rate_per_object = np.std(success_rate_per_object)
            std_success_steps_per_object = np.std(success_steps_per_object)
            std_iou_per_object = np.std(iou_per_object)
            std_coverage_per_object = np.std(coverage_per_object)
            std_chamfer_per_object = np.std(chamfer_per_object)
            std_emd_per_object = np.std(emd_per_object)
            logger.info(f'Std of success rates per object: {std_success_rate_per_object}, success steps: {std_success_steps_per_object}, '
                        f'iou: {std_iou_per_object}, normalized coverage: {std_coverage_per_object}, chamfer distance: {std_chamfer_per_object}, emd: {std_emd_per_object}')

    def vis_metrics(self):
        for object_id in self.object_id_canonical_list:
            logger.info(f'Visualizing metrics of object: {object_id}')
            object_idx = [idx for idx, trial in enumerate(self.data_trials_list)
                          if self.filtered_doc_info_dict_list[trial[0]]['object_id'] == object_id]
            object_iou_per_step_list = [self.iou_per_step_list[idx] for idx in object_idx]
            object_coverage_per_step_list = [self.normalized_coverage_per_step_list[idx] for idx in object_idx]
            object_chamfer_per_step_list = [self.chamfer_per_step_list[idx] for idx in object_idx]
            object_emd_per_step_list = [self.emd_per_step_list[idx] for idx in object_idx]
            for i, (object_iou_per_step, object_coverage_per_step, object_chamfer_per_step, object_emd_per_step) in \
                tqdm.tqdm(enumerate(zip(object_iou_per_step_list, object_coverage_per_step_list, object_chamfer_per_step_list, object_emd_per_step_list)), desc='Visualizing Metrics', total=len(object_iou_per_step_list)):
                # create directory for saving visualization data
                if not osp.exists(self.base_path):
                    os.makedirs(self.base_path)
                
                sns.set_theme()
                sns.set_style("whitegrid")
                sns.set_context("paper", font_scale=1.5)
                color_list = sns.color_palette("Set2", 4)
                fig, ax = plt.subplots(4, 1, figsize=(10, 15))
                ax[0].plot(object_iou_per_step, color=color_list[0])
                ax[0].set_title('IoU')
                ax[0].set_xlabel('Step')
                ax[0].set_xticks(np.arange(0, len(object_iou_per_step)))

                ax[1].plot(object_coverage_per_step, color=color_list[1])
                ax[1].set_title('Normalized Coverage')
                ax[1].set_xlabel('Step')
                ax[1].set_xticks(np.arange(0, len(object_coverage_per_step)))

                ax[2].plot(object_chamfer_per_step, color=color_list[2])
                ax[2].set_title('Chamfer Distance')
                ax[2].set_xlabel('Step')
                ax[2].set_xticks(np.arange(0, len(object_chamfer_per_step)))
                ax[2].set_ylabel('m')

                ax[3].plot(object_emd_per_step, color=color_list[3])
                ax[3].set_title("Earth Mover's Distance")
                ax[3].set_xlabel('Step')
                ax[3].set_xticks(np.arange(0, len(object_emd_per_step)))
                ax[3].set_ylabel('m')

                plt.tight_layout()
                fig.savefig(osp.join(self.base_path, f'{object_id}_{i}.png'))
                plt.close()
                
            object_max_step = max([len(object_iou_per_step) for object_iou_per_step in object_iou_per_step_list])
            object_iou_per_step_list_padding = np.array([object_iou_per_step + [object_iou_per_step[-1]] * (object_max_step - len(object_iou_per_step))
                                                for object_iou_per_step in object_iou_per_step_list])
            object_coverage_per_step_list_padding = np.array([object_coverage_per_step + [object_coverage_per_step[-1]] * (object_max_step - len(object_coverage_per_step))
                                                     for object_coverage_per_step in object_coverage_per_step_list])
            object_chamfer_per_step_list_padding = np.array([object_chamfer_per_step + [object_chamfer_per_step[-1]] * (object_max_step - len(object_chamfer_per_step))
                                                    for object_chamfer_per_step in object_chamfer_per_step_list])
            object_emd_per_step_list_padding = np.array([object_emd_per_step + [object_emd_per_step[-1]] * (object_max_step - len(object_emd_per_step))
                                                    for object_emd_per_step in object_emd_per_step_list])
            object_iou_per_step_mean = np.mean(object_iou_per_step_list_padding, axis=0)
            object_coverage_per_step_mean = np.mean(object_coverage_per_step_list_padding, axis=0)
            object_chamfer_per_step_mean = np.mean(object_chamfer_per_step_list_padding, axis=0)
            object_emd_per_step_mean = np.mean(object_emd_per_step_list_padding, axis=0)
            object_iou_per_step_std = np.std(object_iou_per_step_list_padding, axis=0)
            object_coverage_per_step_std = np.std(object_coverage_per_step_list_padding, axis=0)
            object_chamfer_per_step_std = np.std(object_chamfer_per_step_list_padding, axis=0)
            object_emd_per_step_std = np.std(object_emd_per_step_list_padding, axis=0)
            sns.set_theme()
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.5)
            color_list = sns.color_palette("Set2", 4)
            fig, ax = plt.subplots(4, 1, figsize=(10, 15))
            ax[0].plot(object_iou_per_step_mean, color=color_list[0])
            ax[0].fill_between(np.arange(0, len(object_iou_per_step_mean)), object_iou_per_step_mean - object_iou_per_step_std, object_iou_per_step_mean + object_iou_per_step_std, alpha=0.3, color=color_list[0])
            ax[0].set_title('IoU')
            ax[0].set_xlabel('Step')
            ax[0].set_xticks(np.arange(0, len(object_iou_per_step_mean)))

            ax[1].plot(object_coverage_per_step_mean, color=color_list[1])
            ax[1].fill_between(np.arange(0, len(object_coverage_per_step_mean)), object_coverage_per_step_mean - object_coverage_per_step_std, object_coverage_per_step_mean + object_coverage_per_step_std, alpha=0.3, color=color_list[1])
            ax[1].set_title('Normalized Coverage')
            ax[1].set_xlabel('Step')
            ax[1].set_xticks(np.arange(0, len(object_coverage_per_step_mean)))

            ax[2].plot(object_chamfer_per_step_mean, color=color_list[2])
            ax[2].fill_between(np.arange(0, len(object_chamfer_per_step_mean)), object_chamfer_per_step_mean - object_chamfer_per_step_std, object_chamfer_per_step_mean + object_chamfer_per_step_std, alpha=0.3, color=color_list[2])
            ax[2].set_title('Chamfer Distance')
            ax[2].set_xlabel('Step')
            ax[2].set_xticks(np.arange(0, len(object_chamfer_per_step_mean)))
            ax[2].set_ylabel('m')

            ax[3].plot(object_emd_per_step_mean, color=color_list[3])
            ax[3].fill_between(np.arange(0, len(object_emd_per_step_mean)), object_emd_per_step_mean - object_emd_per_step_std, object_emd_per_step_mean + object_emd_per_step_std, alpha=0.3, color=color_list[3])
            ax[3].set_title("Earth Mover's Distance")
            ax[3].set_xlabel('Step')
            ax[3].set_xticks(np.arange(0, len(object_emd_per_step_mean)))
            ax[3].set_ylabel('m')

            plt.tight_layout()
            fig.savefig(osp.join(self.base_path, f'{object_id}_all.png'))
            plt.close()

            data = {
                'IoU': {
                    'per_step_mean': object_iou_per_step_mean.tolist(),
                    'per_step_std': object_iou_per_step_std.tolist(),
                },
                'Normalized Coverage': {
                    'per_step_mean': object_coverage_per_step_mean.tolist(),
                    'per_step_std': object_coverage_per_step_std.tolist(),
                },
                'Chamfer Distance': {
                    'per_step_mean': object_chamfer_per_step_mean.tolist(),
                    'per_step_std': object_chamfer_per_step_std.tolist(),
                },
                "Earth Mover's Distance": {
                    'per_step_mean': object_emd_per_step_mean.tolist(),
                    'per_step_std': object_emd_per_step_std.tolist(),
                }
            }
            with open(osp.join(self.base_path, f'{object_id}.json'), 'w') as f:
                json.dump(data, f)

if __name__ == '__main__':
    DB_HOST = os.environ.get("DB_HOST")
    DB_PORT = 27017
    DB_USERNAME = os.environ.get("DB_USERNAME")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_URI = f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"

    tag = 'tshirt_short_action14_real_zero_center_new_pipeline_test'
    tag_canonical = 'test'
    
    metric_evaluator = LogMetricEvaluator(mongodb_url=DB_URI,
                                          tag=tag,
                                          tag_canonical=tag_canonical,
                                          step_num_limit=10,
                                          trial_num_limit=20,
                                          debug=False)
    metric_evaluator.calc_mean_metrics(calc_per_object=True)
    metric_evaluator.vis_metrics()