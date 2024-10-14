import copy
import itertools
import os.path as osp
import shutil
from typing import List, Optional, Tuple

import numpy as np
import omegaconf
import py_cli_interaction
import tqdm
import yaml
from omegaconf import OmegaConf

from common.datamodels import AnnotationConfig, AnnotationContext, ActionTypeDef, GeneralObjectState, GarmentSmoothingStyle, \
                                ObjectTypeDef
from tools.run_annotation.functionals import (
    visualize_point_cloud_list_with_points,
    pick_n_points_from_pcd,
)
from tools.run_annotation.io import get_io_module

from loguru import logger


def do_init_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run init annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    Init the annotation context with the log entry.
    """
    context.entry_name = entry_name
    context.annotation_result.annotator = opt.annotator

    # use the side effect and verify the log entry
    _x = list(map(lambda x: list(x), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin))
    return context, None


def do_action_type_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run action type annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB. Then the user types the correct action type (e.g. fling, pick-and-place or fold1).
    """
    __DISPLAY_OPTIONS__ = [
        "fling",
        "pick-and-place",
        "fold1",
        "skip"
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        ActionTypeDef.FLING,
        ActionTypeDef.PICK_AND_PLACE,
        ActionTypeDef.FOLD_1_1,
        ActionTypeDef.DONE,
    ]

    try:
        while True:
            context.console.print("Predicted action: " + str(context.raw_log[opt.raw_log_namespace].action),
                                  style="yellow")
            context.console.print("[instruction] Observe the object and select the action type from the following options:", __DISPLAY_OPTIONS__)
            visualize_point_cloud_list_with_points([context.curr_pcd])

            base = 1
            type_sel = py_cli_interaction.must_parse_cli_sel("Select action type",
                                                             __DISPLAY_OPTIONS__, min=base) - base
            res = __DISPLAY_OPTIONS_MAPPING__[type_sel]

            context.console.print("Your selection is:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                context.annotation_result.action_type = res
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None

def do_general_object_state_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run general object state annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB. Then the user types the correct object state (e.g. organized, disordered or unknown).
    """
    if opt.object_type not in [ObjectTypeDef.TSHIRT_SHORT, ObjectTypeDef.TSHIRT_LONG]:
        logger.info("object type is not a t-shirt, skip general object state annotation")
        return context, None

    __DISPLAY_OPTIONS__ = [
        "organized",
        "disordered",
        "unknown",
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        GeneralObjectState.ORGANIZED,
        GeneralObjectState.DISORDERED,
        GeneralObjectState.UNKNOWN,
    ]

    try:
        while True:
            context.console.print("[instruction] Observe the object and select the state from the following options:", __DISPLAY_OPTIONS__)
            visualize_point_cloud_list_with_points([context.curr_pcd])

            base = 1
            type_sel = py_cli_interaction.must_parse_cli_sel("Select object state",
                                                             __DISPLAY_OPTIONS__, min=base) - base
            res = __DISPLAY_OPTIONS_MAPPING__[type_sel]

            context.console.print("Your selection is:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                context.annotation_result.general_object_state = res
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None

def do_garment_smoothing_style_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run garment smoothing style annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB. Then the user types the correct garment smoothing style (e.g. down, up, left, right, unknow).
    """
    if opt.object_type not in [ObjectTypeDef.TSHIRT_SHORT, ObjectTypeDef.TSHIRT_LONG]:
        logger.info("object type is not a t-shirt, skip garment smoothing style annotation")
        return context, None

    __DISPLAY_OPTIONS__ = [
        "down",
        "up",
        "left",
        "right",
        "unknown",
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        GarmentSmoothingStyle.DOWN,
        GarmentSmoothingStyle.UP,
        GarmentSmoothingStyle.LEFT,
        GarmentSmoothingStyle.RIGHT,
        GarmentSmoothingStyle.UNKNOWN,
    ]

    try:
        # if the garment is disordered, then the smoothing style is unknown
        if context.annotation_result.general_object_state != GeneralObjectState.ORGANIZED:
            logger.info("garment is disordered, then the smoothing style is unknown")
            context.annotation_result.garment_smoothing_style = GarmentSmoothingStyle.UNKNOWN
            return context, None
        
        while True:
            context.console.print("[instruction] Observe the object and select the smoothing style from the following options:", __DISPLAY_OPTIONS__)
            visualize_point_cloud_list_with_points([context.curr_pcd])

            base = 1
            type_sel = py_cli_interaction.must_parse_cli_sel("Select garment smoothing style",
                                                             __DISPLAY_OPTIONS__, min=base) - base
            res = __DISPLAY_OPTIONS_MAPPING__[type_sel]

            context.console.print("Your selection is:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                context.annotation_result.garment_smoothing_style = res
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None

def do_garment_keypoint_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run garment keypoint annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB. Then the user will click on the dense point cloud to annotate the key points for the garment.
    """
    if opt.object_type not in [ObjectTypeDef.TSHIRT_SHORT, ObjectTypeDef.TSHIRT_LONG]:
        logger.info("object type is not a t-shirt, skip garment keypoint annotation")
        return context, None

    keypoint_num = opt.keypoint_num
    assert keypoint_num == 4, "Only support 4 keypoints for now."
    res = [None] * keypoint_num
    __KEYPOINT_NAMES__ = [
        "left shoulder",
        "right shoulder",
        "left waist",
        "right waist",
    ]
    try:
        if context.annotation_result.general_object_state != GeneralObjectState.ORGANIZED:
            logger.info("garment is disordered, then the keypoints annotation are not needed")
            return context, None
        
        while True:
            context.console.print(f"[instruction] Select ideal keypoints {', '.join(__KEYPOINT_NAMES__)} in order")
            pts, _, err = pick_n_points_from_pcd(context.curr_pcd, keypoint_num)
            if err is not None:
                context.console.print(err, style="red")
                continue
            res = pts

            visualize_point_cloud_list_with_points([context.curr_pcd], points=pts)
            context.console.print("Your selection is:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                context.annotation_result.garment_keypoints = res
                break
            else:
                continue
            
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None


def do_auto_action_type_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run automatically action type annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    Use the logged action type as the annotation result because it is obtained from manual operation.
    """
    action_type = ActionTypeDef.from_string(context.raw_log[opt.raw_log_namespace].action.type)
    context.console.print("Automatic action type annotation: ", action_type, style="blue")
    context.annotation_result.action_type = action_type
    return context, None


def do_action_pose_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run action pose annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB and two pair of possible grasp-points predicted by the policy model. The user will click on the dense point cloud to annotate the correct pick points or place points for this action.
    """
    res = [None] * 4
    try:
        while True:
            if context.annotation_result.action_type in [ActionTypeDef.FOLD_1_1]:
                context.console.print(
                    "[instruction] Select ideal poses, click 4 points in order: left_pick, right_pick, left_place, right_place")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 4)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1], res[2], res[3] = pts
            elif context.annotation_result.action_type in [ActionTypeDef.PICK_AND_PLACE]:
                context.console.print(
                    "[instruction] Select ideal poses, click 4 points in order: left_pick, right_pick, left_place, right_place")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 4)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1], res[2], res[3] = pts
            elif context.annotation_result.action_type in [ActionTypeDef.FLING]:
                context.console.print(
                    "[instruction] Select ideal poses, click 2 points in order: left_pick, right_pick")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1] = pts

            elif context.annotation_result.action_type in [ActionTypeDef.SWEEP]:
                context.console.print(
                    "[instruction] Select ideal poses, click 2 points in order: start, end")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2, wiper_width=0.1)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[2] = pts
                res[1], res[3] = pts

            elif context.annotation_result.action_type in [ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                context.console.print(
                    "[instruction] Select ideal poses, click 2 points in order: start, end")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[2] = pts
                res[1], res[3] = pts

            elif context.annotation_result.action_type in [ActionTypeDef.DONE]:
                return context, None

            else:
                return context, Exception(NotImplementedError)

            visualize_point_cloud_list_with_points([context.curr_pcd], points=pts)
            context.console.print("Your selection is:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                context.annotation_result.action_poses = list(
                    map(
                        lambda x: np.array([x[0], x[1], x[2], 0, 0, 0]) if x is not None else np.zeros(shape=(6,),
                                                                                                       dtype=float),
                        res
                    )
                )
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None


def do_grasp_ranking_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run grasp ranking annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    Grasp-point Ranking Annotation: This step is only for fling action!!! The annotation software will repeat the following steps for K times:

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB and two pair of possible grasp-points predicted by the policy model. We denote the the first pair of points as P1, and the second pair of points as P2.
    The volunteer gives 4 possible ranking annotation:

    P1 > P2 (P1 is better)
    P1 < P2 (P2 is better)
    P1 = P2 (equally good)
    Not comparable (hard to distinguish for humans).
    """

    __DISPLAY_OPTIONS__ = [
        "P1 = P2",
        "P1 > P2",
        "P1 < P2",
        "Not comparable"
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        2,
        0,
        1,
        3
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
    __NOT_COMPARABLE_INDEX__ = 3
    __REWARD_THRESH_INIT__ = -0.2
    __REWARD_THRESH_FINAL__ = -0.5
    __REWARD_NEGATIVE_INF__ = -1000.0
    __USE_DYNAMIC_THRESH__ = True
    __REWARD_TOP_RATIO__ = 0.2
    __MIN_NUM_CANDIDATES_PAIR__ = 5
    __ALWAYS_SELECT_TOP_CANDIDATES__ = True
    __MIN_NUM_CANDIDATES_PAIR_PRE_FILTERING__ = 10

    candidates = list(map(lambda x: list(x), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin))

    if opt.annotation_type == "real_finetune":
        __UNIQUE_THRESH__ = 0.02
        
        if __USE_DYNAMIC_THRESH__:
            reward_matrix = np.array(context.raw_log[opt.raw_log_namespace].prediction_info.reward.virtual).mean(axis=-1)
            assert len(reward_matrix.shape) == 2
            num_candidates = reward_matrix.shape[0]  # K
            # make sure that the same point is not selected
            self_idxs = np.arange(num_candidates)
            reward_matrix[self_idxs, self_idxs] = __REWARD_NEGATIVE_INF__
            flatten_pair_score = reward_matrix.reshape((-1,))  # (K*K, )
            sorted_pair_idxs = np.argsort(flatten_pair_score)[::-1]  # (K*K, )
            thresh_pair_idx = sorted_pair_idxs[int(len(flatten_pair_score) * __REWARD_TOP_RATIO__)]
            idx1 = thresh_pair_idx // num_candidates
            idx2 = thresh_pair_idx % num_candidates
            _reward_thresh = reward_matrix[idx1, idx2]
        else:
            _reward_thresh = __REWARD_THRESH_INIT__

        while True:
            candidates_pair = list(zip(*np.where(np.triu(
                np.array(context.raw_log[opt.raw_log_namespace].prediction_info.reward.virtual).squeeze() - _reward_thresh,
                k=1) >= 0)))
            # list(itertools.combinations(range(0, len(candidates)), 2))
            if len(candidates_pair) < __MIN_NUM_CANDIDATES_PAIR__ and _reward_thresh > __REWARD_THRESH_FINAL__:
                _reward_thresh -= 0.1
            else:
                break
        
        if len(candidates_pair) < __MIN_NUM_CANDIDATES_PAIR__:
            context.console.print("too many bad grasp points, fallback to traditional algorithm",
                                style="red")
            all_candidates_pair = list(itertools.combinations(range(0, len(candidates)), 2))
            if __ALWAYS_SELECT_TOP_CANDIDATES__:
                res_num = __MIN_NUM_CANDIDATES_PAIR_PRE_FILTERING__ - len(candidates_pair)
                np.random.shuffle(all_candidates_pair)
                candidates_pair.extend(all_candidates_pair[:res_num])
            else:
                candidates_pair = all_candidates_pair
        candidates_pair = list(map(lambda x: (int(x[0]), int(x[1])), candidates_pair))
        
    elif opt.annotation_type == "new_pipeline_finetune":
        __UNIQUE_THRESH__ = 0.0
        candidates_pair = list(zip(range(0, len(candidates) - 1, 2), range(1, len(candidates), 2)))
        
    else:
        raise ValueError(f"unknown annotation type: {opt.annotation_type}")
    

    def is_grasp_point_safe(points: np.ndarray):
        d = np.linalg.norm(points[0][:3] - points[1][:3])
        return float(d) > 0.15  # TODO: remove magic number

    def is_grasp_point_unique(s, c, points):
        ret = True
        for index in s:
            curr_points = np.array([c[x] for x in index[:2]])
            for i in range(2):
                for j in range(2):
                    if 1e-4 < np.linalg.norm(points[i][:3] - curr_points[j][:3]) < __UNIQUE_THRESH__:  # TODO: remove magic number
                        ret = False
            if not ret:
                break
        return ret
    
    combos = list(itertools.combinations(candidates_pair, 2))
    if opt.K > len(combos):
        return context, Exception(f"trying to run K comparison with {len(combos)} is illegal")

    combo_indices = np.random.permutation(np.arange(0, len(combos)))

    if context.annotation_result == ActionTypeDef.DONE:
        return context, None
    elif context.annotation_result.action_type not in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
        selected_grasp_point_indices: List[Optional[List[int]]] = [[0, 0, 0, 0] for _ in range(opt.K)]
        grasp_point_rankings = [__NOT_COMPARABLE_INDEX__] * opt.K
    else:
        ordering = True if context.annotation_result.action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE] else False
        while True:
            selected_grasp_point_indices: List[Optional[List[
                int]]] = []  # np.array(combos)[selected_indices_non_result].reshape(len(selected_indices_non_result), -1)
            grasp_point_rankings: List[Optional[int]] = []
            with tqdm.tqdm(total=opt.K) as pbar:
                for ranking_idx, compare_idx in enumerate(combo_indices):
                    left_op, right_op = combos[compare_idx]
                    left_points_np, right_points_np = np.array([candidates[x] for x in left_op]), np.array(
                        [candidates[x] for x in right_op])

                    if len(combo_indices) - ranking_idx + pbar.n > opt.K:
                        if not (is_grasp_point_safe(left_points_np) and is_grasp_point_safe(right_points_np)):
                            continue

                        if not (
                                is_grasp_point_unique(
                                    selected_grasp_point_indices,
                                    candidates,
                                    left_points_np
                                ) and
                                is_grasp_point_unique(
                                    selected_grasp_point_indices,
                                    candidates,
                                    right_points_np
                                )
                        ):
                            continue
                    else:
                        context.console.print("insufficient grasp pairs detected, disable filtering",
                                              style="red")

                    context.console.print("\ngrasp_point_pair: ", combos[compare_idx], style="yellow")
                    # Generate copy of point cloud
                    left_pcd, right_pcd = copy.deepcopy(context.curr_pcd), copy.deepcopy(context.curr_pcd)

                    # Move the point cloud for a distance
                    right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
                    right_points_np[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__

                    all_points = np.vstack([left_points_np[..., :3], right_points_np[..., :3]])
                    context.console.print(
                        "[instruction] Please observe and compare P1(left) to P2(right). Which one is better?")
                    visualize_point_cloud_list_with_points([left_pcd, right_pcd], points=all_points, zoom=0.25, points_group_size=2, ordering=ordering)

                    
                    base = 1
                    _sel = py_cli_interaction.must_parse_cli_sel("Ranking: ", __DISPLAY_OPTIONS__, min=base) - base
                    _res = __DISPLAY_OPTIONS_MAPPING__[_sel]

                    selected_grasp_point_indices.append(list(left_op) + list(right_op))
                    grasp_point_rankings.append(_res)
                    pbar.update()

                    if pbar.n >= opt.K:
                        break

            context.console.print("Your selection is:",
                                  [__DISPLAY_OPTIONS_MAPPING_INV__[x] for x in grasp_point_rankings], style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:
                break
            else:
                continue

    context.annotation_result.grasp_point_rankings = grasp_point_rankings
    context.annotation_result.selected_grasp_point_indices = selected_grasp_point_indices

    return context, None

def do_grasp_ranking_sort_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    '''
    run grasp ranking annotation with sorting
    
    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any
        
    This step is only for fling action. The annotation software will iteratively:
        1. Display all the remaining pair of grasp-points predicted by the policy model, with a number denoting each pair of points.
        2. Ask the volunteer to select the top 1 pair of grasp-points that are the best.
        3. Remove the selected pair of grasp-points from the display.
    until the volunteer reckons the remaining pairs are not comparable.
    '''
    __FIRST_BETTER_INDEX__ = 0
    __NOT_COMPARABLE_INDEX__ = 3
    if context.annotation_result == ActionTypeDef.DONE:
        return context, None
    elif context.annotation_result.action_type not in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
        return context, None
    else:
        ordering = True if context.annotation_result.action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE] else False
        wiper_width = 0.1 if context.annotation_result.action_type == ActionTypeDef.SWEEP else None
        while True:
            candidates = list(map(lambda x: list(x), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin))
            
            if opt.annotation_type == "new_pipeline_finetune_sort":
                candidates_pair = list(zip(range(0, len(candidates) - 1, 2), range(1, len(candidates), 2)))
                
            else:
                raise ValueError(f"unknown annotation type: {opt.annotation_type}")
            
            num_pairs = len(candidates_pair)
            num_remaining_pairs = num_pairs
            
            pair_rest = list(range(num_pairs))
            pair_sorted = []
            pcd = copy.deepcopy(context.curr_pcd)
            base = 1
            while num_remaining_pairs > 0:
                context.console.print(f"\n[instruction] Observe the pairs of grasp points and remember the number of the best pair")
                points_remaining = np.array([candidates[x * 2 + lr][:3] for x in pair_rest for lr in range(2)])
                labels = [str(x + base) for x in pair_rest for _ in range(2)]
                visualize_point_cloud_list_with_points([pcd], points=points_remaining, labels=labels, points_group_size=2, ordering=ordering, wiper_width=wiper_width)
                confirm_complete = py_cli_interaction.must_parse_cli_bool("Do you confirm that the remaining points are hard to distinguish?", default_value=False)
                if confirm_complete:
                    break
                while True:
                    _sel = py_cli_interaction.must_parse_cli_int("\nEnter the number of the best pair of grasp points", min=1, max=num_pairs)
                    _sel -= base
                    if _sel in pair_rest:
                        break
                    context.console.print(f"Invalid selection, please observe and re-enter:")
                    visualize_point_cloud_list_with_points([pcd], points=points_remaining, labels=labels, points_group_size=2, ordering=ordering, wiper_width=wiper_width)
                pair_sorted.append(_sel)
                pair_rest.remove(_sel)
                num_remaining_pairs -= 1
                
            context.console.print("Your selection is:", [x + base for x in pair_sorted], style="blue")
            context.console.print("The remaining is:", [x + base for x in pair_rest], style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("Confirm?", default_value=True)
            if confirm:    
                context.annotation_result.grasp_point_pair_sorted = pair_sorted
                context.annotation_result.grasp_point_pair_rest = pair_rest
                
                # generate K pairs
                grasp_point_rankings = []
                selected_grasp_point_indices = []
                # first the tops and the unsorted
                for i in pair_sorted:
                    for j in pair_rest:
                        if len(grasp_point_rankings) >= opt.K:
                            break
                        grasp_point_rankings.append(__FIRST_BETTER_INDEX__)
                        selected_grasp_point_indices.append([*candidates_pair[i], *candidates_pair[j]])
                            
                # then if it is not enough, generate sorting from the sorted pairs 
                for dist in range(len(pair_sorted) - 1, 0, -1):
                    for i in range(len(pair_sorted) - dist):
                        if len(grasp_point_rankings) >= opt.K:
                            break
                        grasp_point_rankings.append(__FIRST_BETTER_INDEX__)
                        selected_grasp_point_indices.append([*candidates_pair[pair_sorted[i]], *candidates_pair[pair_sorted[i + dist]]])
                        
                # then if it is still not enough, generate sorting from the unsorted pairs
                while len(grasp_point_rankings) < opt.K:
                    grasp_point_rankings.append(__NOT_COMPARABLE_INDEX__)
                    selected_grasp_point_indices.append([0, 0, 0, 0])
                    
                context.annotation_result.grasp_point_rankings = grasp_point_rankings
                context.annotation_result.selected_grasp_point_indices = selected_grasp_point_indices

                return context, None

def do_fling_automatic_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """do fling automatic annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context as completed

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any

    We need to add Step 4 for fling action.
    In Step 2, the GT pick points from human annotator could be used for oracle ranking, because they are probably better than any predicted grasp-points from the AI model. So we need to perform Step 4 after Step 3:

    The annotation software displays two identical masked point cloud (before the action) with RGB, then shows GT pick points on the left side, and all other predicted pick points on the right side.
    The user should give 2 possible annotation:
    Case 1: The GT pick points are better than any other predicted point pairs.
    Case 2: The GT pick points are not comparable with other predicted point pairs (usually happpens under very disordered object states with multiple good candidates).
    Append the GT pick points into virtual_posses.predictions of the metadata.
    Append the automatic ranking annotation into annotation.selected_grasp_point_indices and annotation.grasp_point_rankings.
    Case 1: All the additional ranking results are P1 > P2 (label 0).
    Case 2: All the additional ranking results are P1 not comparbale with P2 (label 3).
    """
    __DISPLAY_POINTCLOUD_OFFSET__ = np.array([
        1.5, 0., 0.
    ])
    if context.annotation_result.action_type in [ActionTypeDef.FLING, ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
        # Generate copy of point cloud
        left_pcd, right_pcd = copy.deepcopy(context.curr_pcd), copy.deepcopy(context.curr_pcd)

        if context.annotation_result.action_type == ActionTypeDef.FLING:
            left_points_np = np.array(context.annotation_result.action_poses[:2])
        elif context.annotation_result.action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
            left_points_np = np.array(context.annotation_result.action_poses[::2])

        right_points_np = np.array(list(
            map(lambda x: np.array(list(x)), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin)))

        # Move the point cloud for a distance
        right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
        right_points_np[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__

        all_points = np.vstack([left_points_np[..., :3], right_points_np[..., :3]])

        context.console.print("[instruction] Please observe and compare the left and right pairs")
        ordering = True if context.annotation_result.action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE] else False
        wiper_width = 0.1 if context.annotation_result.action_type == ActionTypeDef.SWEEP else None
        visualize_point_cloud_list_with_points([left_pcd, right_pcd], points=all_points, zoom=0.25, points_group_size=2, ordering=ordering, wiper_width=wiper_width)

        context.annotation_result.fling_gt_is_better_than_rest = py_cli_interaction.must_parse_cli_bool(
            "Is GT better than any other predicted pairs?")
        context.console.print("Your selection is: ", context.annotation_result.fling_gt_is_better_than_rest,
                              style="blue")

    else:
        context.annotation_result.fling_gt_is_better_than_rest = None

    return context, None

def do_finalize(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """do finalize

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context as completed

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any
    """
    annotated_log = copy.deepcopy(context.raw_log)

    with omegaconf.open_dict(annotated_log):
        __FLING_GT_IS_BETTER_THAN_REST_MAPPING__ = {True: 0, False: 3, None: 3}
        _k = len(annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin)
        if opt.annotation_type == "real_finetune" or opt.annotation_type == "new_pipeline_supervised" \
            or opt.annotation_type == "new_pipeline_supervised_classification_detection":
            candidates_pair = list(itertools.combinations(range(0, _k), 2))
        elif opt.annotation_type == "new_pipeline_finetune" or opt.annotation_type == "new_pipeline_finetune_sort":
            # use adjacent pairs as candidates
            candidates_pair = list(zip(range(0, _k - 1, 2), range(1, _k, 2)))
        else:
            raise ValueError(f"unknown annotation type: {opt.annotation_type}")

        if context.annotation_result.fling_gt_is_better_than_rest is not None:
            # Insert points to data
            annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin += list(
                map(lambda x: omegaconf.ListConfig(x.tolist()), context.annotation_result.action_poses[:2]))
            __k = list(range(_k, len(annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin)))

            # add generated result
            context.annotation_result.selected_grasp_point_indices.extend([[*__k, *x] for x in candidates_pair])
            context.annotation_result.grasp_point_rankings.extend([__FLING_GT_IS_BETTER_THAN_REST_MAPPING__[
                                                                       context.annotation_result.fling_gt_is_better_than_rest]] * len(
                candidates_pair))

        else:
            annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin += list(
                map(lambda x: omegaconf.ListConfig(x.tolist()), np.zeros(shape=(2, 6))))

            context.annotation_result.selected_grasp_point_indices.extend([[0, 0, 0, 0] for _ in candidates_pair])
            context.annotation_result.grasp_point_rankings.extend([__FLING_GT_IS_BETTER_THAN_REST_MAPPING__[
                                                                       context.annotation_result.fling_gt_is_better_than_rest]] * len(
                candidates_pair))

        annotation_dict = context.annotation_result.to_dict()
        annotated_log[opt.raw_log_namespace].annotation = omegaconf.DictConfig(annotation_dict)

    err = get_io_module(opt).move_for_backup(entry_name)
    if err is not None:
        context.console.print("Failed", style="red")
        return context, err

    err = get_io_module(opt).write_annotation(entry_name, annotation_dict, annotated_log)
    if err is not None:
        context.console.print("Failed", style="red")
        return context, err

    return context, None


def do_existing_action_pose_visualization(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """do existing action pose visualization

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context as completed

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any

    The annotation software displays the masked point cloud (before the action) with RGB, then shows the existing GT pick points.

    """
    context_action_type = ActionTypeDef.from_string(context.raw_log[opt.raw_log_namespace].action.type)
    if context_action_type == ActionTypeDef.FLING:
        # Generate copy of point cloud
        pcd = copy.deepcopy(context.curr_pcd)
        points_np = np.array([
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.begin.left,
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.begin.right
            ])
        
        context.annotation_result.multiple_action_poses[0][0] = points_np[0]
        context.annotation_result.multiple_action_poses[0][1] = points_np[1]
        
        context.annotation_result.action_poses[0] = points_np[0]
        context.annotation_result.action_poses[1] = points_np[1]
        
        points_np = points_np[..., :3]
        
        context.console.print("[instruction] Observe the existing pick points")
        visualize_point_cloud_list_with_points([pcd], points=points_np)
    
    elif context_action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
        wiper_width = 0.1 if context.annotation_result.action_type == ActionTypeDef.SWEEP else None
        # Generate copy of point cloud
        pcd = copy.deepcopy(context.curr_pcd)
        points_np = np.array([
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.begin.left,
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.end.left
            ])
        if np.all(points_np[0] == 0) or np.all(points_np[1] == 0):
            points_np = np.array([
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.begin.right,
                context.raw_log[opt.raw_log_namespace].pose_virtual.gripper.end.right
            ])
        
        context.annotation_result.multiple_action_poses[0][0] = \
        context.annotation_result.multiple_action_poses[0][1] = points_np[0]
        context.annotation_result.multiple_action_poses[0][2] = \
        context.annotation_result.multiple_action_poses[0][3] = points_np[1]
        
        context.annotation_result.action_poses[0] = \
        context.annotation_result.action_poses[1] = points_np[0]
        context.annotation_result.action_poses[2] = \
        context.annotation_result.action_poses[3] = points_np[1]
        
        points_np = points_np[..., :3]
        
        context.console.print("[instruction] Observe the existing pick points")
        visualize_point_cloud_list_with_points([pcd], points=points_np, points_group_size=2, ordering=True, wiper_width=wiper_width)
    
    else:
        logger.info(f"unsupported action type: {context_action_type}, skip visualization")
    
    return context, None


def do_multiple_action_pose_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run extra action pose annotation, which is used for the supervised stage of the new pipeline

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB and two pair of possible grasp-points predicted by the policy model. The user will click on the dense point cloud to annotate the correct pick points or place points for this action.
    """
    multiple_action_poses = []
    try:
        wiper_width = 0.1 if context.annotation_result.action_type == ActionTypeDef.SWEEP else None
        while True:
            if opt.multi_pose_num > 0:
                if len(multiple_action_poses) >= opt.multi_pose_num - 1:
                    next_pose = False
                else:
                    context.console.print(f"{opt.multi_pose_num - 1 - len(multiple_action_poses)} more poses to annotate")
                    next_pose = True
            else:
                next_pose = py_cli_interaction.must_parse_cli_bool("Are there any other potential poses?", default_value=True)
            if not next_pose:
                context.annotation_result.multiple_action_poses += multiple_action_poses
                break
            res = [None] * 4
            context_action_type = ActionTypeDef.from_string(context.raw_log[opt.raw_log_namespace].action.type)
            if context_action_type in [ActionTypeDef.FLING]:
                context.console.print(
                    "[instruction] Select another ideal poses, click 2 points in order: left_pick, right_pick")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2, wiper_width=wiper_width)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1] = pts
            
            elif context.annotation_result.action_type in [ActionTypeDef.SWEEP, ActionTypeDef.SINGLE_PICK_AND_PLACE]:
                context.console.print(
                    "[instruction] Select ideal poses, click 2 points in order: left_pick, right_pick")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2, wiper_width=wiper_width)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[2] = pts
                res[1], res[3] = pts

            else:
                logger.info(f"unsupported action type: {context_action_type}, skip annotation")
                return context, None

            visualize_point_cloud_list_with_points([context.curr_pcd], points=pts, points_group_size=2, ordering=True)
            context.console.print("Your selection is:", res, style="blue")
            multiple_action_poses.append(list(
                    map(
                        lambda x: np.array([x[0], x[1], x[2], 0, 0, 0]) if x is not None else np.zeros(shape=(6,),
                                                                                                    dtype=float),
                        res
                    )
                )
            )
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None
