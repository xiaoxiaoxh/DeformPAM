import copy
from typing import Optional, List, Dict

import numpy as np
import py_cli_interaction
from loguru import logger
from rich.console import Console

from common.datamodels import AnnotationConfig, AnnotationContext, AnnotationFlag, AnnotationResult, ActionTypeDef
from functionals import (
    pick_n_points_from_pcd,
    visualize_point_cloud_list_with_points,
)
from tools.run_annotation.io import get_io_module
from tools.run_annotation.operations import do_init_annotation, \
    do_general_object_state_annotation, do_garment_smoothing_style_annotation, do_garment_keypoint_annotation, \
    do_action_type_annotation, do_action_pose_annotation, \
    do_grasp_ranking_annotation, do_grasp_ranking_sort_annotation, \
    do_fling_automatic_annotation, do_finalize, \
    do_existing_action_pose_visualization, do_auto_action_type_annotation, do_multiple_action_pose_annotation

from functools import partial

def tui_interaction(opt: AnnotationConfig, unprocessed_log_entries: List[str]) -> Optional[Exception]:
    """tui interaction"""
    if opt.annotation_type == 'real_finetune':
        __ACTIONS__ = [
            do_init_annotation,
            do_action_type_annotation,
            do_action_pose_annotation,
            do_grasp_ranking_annotation,
            do_fling_automatic_annotation,
            do_finalize
        ]
    elif opt.annotation_type == 'new_pipeline_supervised':
        __ACTIONS__ = [
            do_init_annotation,
            do_auto_action_type_annotation,
            do_general_object_state_annotation,
            do_garment_smoothing_style_annotation,
            do_garment_keypoint_annotation,
            do_existing_action_pose_visualization,
            do_multiple_action_pose_annotation,
            do_finalize
        ]
    elif opt.annotation_type == 'new_pipeline_supervised_classification_detection':
        __ACTIONS__ = [
            do_init_annotation,
            do_auto_action_type_annotation,
            do_general_object_state_annotation,
            do_garment_smoothing_style_annotation,
            do_garment_keypoint_annotation,
            do_finalize
        ]
    elif opt.annotation_type == 'new_pipeline_finetune':
        __ACTIONS__ = [
            do_init_annotation,
            do_auto_action_type_annotation,
            do_action_pose_annotation,
            do_grasp_ranking_annotation,
            do_fling_automatic_annotation,
            do_finalize
        ]
    elif opt.annotation_type == 'new_pipeline_finetune_sort':
        __ACTIONS__ = [
            do_init_annotation,
            do_auto_action_type_annotation,
            do_action_pose_annotation,
            do_grasp_ranking_sort_annotation,
            do_fling_automatic_annotation,
            do_finalize
        ]
    else:
        raise ValueError(f"unknown annotation type: {opt.annotation_type}")
    # vis = get_live_visualizer()
    logger.info(f"found {len(unprocessed_log_entries)} entries")
    exit_flag = False

    io_module = get_io_module(opt)

    for entry in unprocessed_log_entries:
        # vis.clear_geometries()
        if io_module.acquire_annotation_lock(entry) is not None:
            logger.warning("failed to acquire lock, skipping")
            continue

        res, err = io_module.get_log_processed_flag(entry)
        if res in [AnnotationFlag.COMPLETED.value, AnnotationFlag.CORRUPTED.value]:
            logger.info(f"already annotated/corrupted ({res}), skipping")
            io_module.release_annotation_lock(entry)
            continue

        ctx = AnnotationContext(io_module)
        ctx.console = Console()
        ctx.console.print("annotating: ", entry)
        # ctx.vis = vis

        entry_ok = True
        for action in __ACTIONS__:
            try:
                ctx, err = action(opt, entry, ctx)
                if err is not None:
                    logger.error(f'error: {err}')
                    entry_ok = False
                    if err.args[0] != KeyboardInterrupt:
                        io_module.set_log_processed_flag(entry, AnnotationFlag.CORRUPTED)
                    else:
                        exit_flag = True
                    break
            except KeyboardInterrupt:
                io_module.release_annotation_lock(entry)
                return Exception(KeyboardInterrupt)

            except Exception as e:
                logger.exception(e)
                entry_ok = False
                io_module.release_annotation_lock(entry)
                io_module.set_log_processed_flag(entry, AnnotationFlag.CORRUPTED)
                break

        if entry_ok:
            # Save log entry from ctx
            io_module.set_log_processed_flag(entry, AnnotationFlag.COMPLETED)
            io_module.release_annotation_lock(entry)
        else:
            if exit_flag:
                io_module.release_annotation_lock(entry)
                return Exception(KeyboardInterrupt)
            else:
                continue

    # vis.destroy_window()
    # vis.close()
    return None


def random_verify(opt: AnnotationConfig, processed_log_entries: List[str]) -> Optional[Exception]:
    """random verify"""
    __DISPLAY_OPTIONS__ = [
        "P1 = P2",
        "P1 > P2",
        "P1 < P2",
        "Not comparable"
    ]
    __DISPLAY_OPTIONS_MAPPING_INV__ = [
        1,
        2,
        0,
        3
    ]
    __DISPLAY_POINTCLOUD_OFFSET__ = np.array([
        1.5, 0., 0.
    ])
    __DISPLAY_NUM_VIRTUAL_POINTS__ = 2

    exit_flag = False
    io_module = get_io_module(opt)

    import datetime
    all_timestamps: Dict[datetime.datetime, str] = {
        datetime.datetime.fromtimestamp(int(x[0].split('.')[1]) / 1000) \
            : x[1] for x in processed_log_entries
    }
    all_dates = set(list(map(lambda x: x.strftime('%Y-%m-%d'), all_timestamps.keys())))
    date_timestamp_map = {
        k: list(filter(lambda x: x.strftime('%Y-%m-%d') == k, all_timestamps)) for k in all_dates
    }
    date_candidates = sorted(list(date_timestamp_map.keys()))
    date_sel = py_cli_interaction.must_parse_cli_sel(msg="select date", candidates=date_candidates)
    selected_log_entries = [all_timestamps[k] for k in date_timestamp_map[date_candidates[date_sel]]]

    index = np.random.permutation(np.arange(0, len(selected_log_entries)))

    console = Console()
    for entry in selected_log_entries:
        if io_module.acquire_annotation_lock(entry) is not None:
            logger.warning("failed to acquire lock, skipping")
            continue

        res, err = io_module.get_log_processed_flag(entry)
        if res != AnnotationFlag.COMPLETED.value:
            logger.info(f"not yet annotated ({res}), skipping")
            io_module.release_annotation_lock(entry)
            continue

        logger.info(f'exam {entry}')
        ctx = AnnotationContext(io_module)
        try:
            ctx, err = do_init_annotation(opt, entry, ctx)
            if err is not None:
                logger.error(f'error: {err}')
                if err.args[0] == KeyboardInterrupt:
                        exit_flag = True
            
            annotation_dicts, err = io_module.get_annotation(entry)
            if err is not None:
                logger.error(f'error: {err}')
        
        except KeyboardInterrupt:
            io_module.release_annotation_lock(entry)
            return Exception(KeyboardInterrupt)
       
        except Exception as e:
            logger.exception(e)

        finally:
            io_module.release_annotation_lock(entry)

        if exit_flag:
            return Exception(KeyboardInterrupt)

        for annotator, annotation_dict in annotation_dicts.items():
            logger.info(f'check annotation of {annotator.split(".")[0]}')
            try:
                annotation = AnnotationResult()
                annotation.from_dict(annotation_dict)
            except Exception:
                logger.error('failed to get annotation')
                continue

            candidates = np.array(ctx.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin)
            console.print("action_type:", annotation.action_type, style="blue")
            if (opt.annotation_type == "real_finetune" or opt.annotation_type == "new_pipeline_supervised" or opt.annotation_type == "new_pipeline_finetune") \
                and annotation.action_type == ActionTypeDef.FLING:
                console.print("action_poses during execution:", annotation.action_poses, style="blue")
                visualize_point_cloud_list_with_points([ctx.curr_pcd], np.array(annotation.action_poses)[:, :3])
            
            if opt.annotation_type == "new_pipeline_supervised" and annotation.action_type == ActionTypeDef.FLING:
                for i in range(1, opt.multi_pose_num):
                    console.print(f"multiple_action_poses[{i}]:", annotation.multiple_action_poses[i], style="blue")
                    visualize_point_cloud_list_with_points([ctx.curr_pcd], np.array(annotation.multiple_action_poses[i])[:, :3])
            
            if (opt.annotation_type == "new_pipeline_supervised_classification_detection"):
                console.print("general_object_state:", annotation.general_object_state, style="blue")
                console.print("garment_smoothing_style:", annotation.garment_smoothing_style, style="blue")
                visualize_point_cloud_list_with_points([ctx.curr_pcd], np.array(annotation.garment_keypoints))
                
            if (opt.annotation_type == "real_finetune" or opt.annotation_type == "new_pipeline_finetune") and annotation.action_type == ActionTypeDef.FLING:
                for i in range(opt.K):
                    console.print(f"grasp_point_rankings[{i}]:", __DISPLAY_OPTIONS__[__DISPLAY_OPTIONS_MAPPING_INV__[annotation.grasp_point_rankings[i]]], style="blue")
                    point_indexes = annotation.selected_grasp_point_indices[i]
                    left_op, right_op = point_indexes[:2], point_indexes[2:]
                    left_points_np, right_points_np = candidates[left_op], candidates[right_op]

                    # Generate copy of point cloud
                    left_pcd, right_pcd = copy.deepcopy(ctx.curr_pcd), copy.deepcopy(ctx.curr_pcd)

                    # Move the point cloud for a distance
                    right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
                    right_points_np[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__

                    all_points = np.vstack([left_points_np[..., :3], right_points_np[..., :3]])
                    visualize_point_cloud_list_with_points([left_pcd, right_pcd], points=all_points, zoom=0.25)

                # visualize the all grasp points and annotated grasp points, print yes or no
                console.print("fling_gt_is_better_than_rest:", annotation.fling_gt_is_better_than_rest, style="blue")
                left_pcd, right_pcd, right_candidates = copy.deepcopy(ctx.curr_pcd), copy.deepcopy(ctx.curr_pcd), copy.deepcopy(candidates)
                right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
                right_candidates[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__
                right_candidates = right_candidates[:-__DISPLAY_NUM_VIRTUAL_POINTS__]

                visualize_point_cloud_list_with_points(
                    [left_pcd, right_pcd],
                    np.concatenate(
                        [np.array(annotation.action_poses)[:, :3], right_candidates[:, :3]]
                    ),
                    point_colors=[0] * 4 + [1 + i for i in range(len(right_candidates))],
                    zoom=0.25
                )

    return None


def display(opt: AnnotationConfig, log_entries: List[str]) -> Optional[Exception]:
    """random verify"""

    console = Console()
    io_module = get_io_module(opt)

    logger.info(f"exam {len(log_entries)} entries")
    for entry in log_entries:
        logger.info(f'exam {entry}')
        ctx = AnnotationContext(io_module)
        try:
            ctx, err = do_init_annotation(opt, entry, ctx)
        except:
            n = py_cli_interaction.must_parse_cli_int("input n=")
            grasp_points, _, _ = pick_n_points_from_pcd(ctx.curr_pcd, n)
            visualize_point_cloud_list_with_points([ctx.curr_pcd], grasp_points[:, :3])
            continue

        candidates = np.array(ctx.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin)
        decisions = np.array(ctx.raw_log[opt.raw_log_namespace].decision.begin)
        action_type_str = ctx.raw_log[opt.raw_log_namespace].action.type
        console.print("action_type:", action_type_str, style="blue")

        decision_is_invalid = -1 in decisions
        if decision_is_invalid:
            n = py_cli_interaction.must_parse_cli_int("input n=")
            grasp_points, _, _ = pick_n_points_from_pcd(ctx.curr_pcd, n)
        else:
            grasp_points = candidates[decisions]
        visualize_point_cloud_list_with_points([ctx.curr_pcd], grasp_points[:, :3])

        if action_type_str == "fling":
            pass
        else:
            pass

    return None
