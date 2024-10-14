import time
import math
import numpy as np
from common.datamodels import GeneralDualArmExecutionCheckingContext
from scipy.spatial.transform import Rotation as R
import itertools

UPDATE_ACTUATOR = False

dummy_jps = [0.0] * 14

from loguru import logger

def check_execution_all(context: GeneralDualArmExecutionCheckingContext, checker_list) -> bool:
    """
    Check the execution of the poses and waypoints with a list of checkers.
    """
    for checker in checker_list:
        if not context.execution_result.overall_success:
            logger.debug(f"Action not executable: {context.execution_result.error_types}")
            break
        checker(context)
    if not context.execution_result.overall_success:
        logger.debug(f"Action not executable: {context.execution_result.error_types}")
    return context.execution_result.overall_success

def setJointSpaceState(actuator, left_joints_value, right_joints_value, delay=1.0):
    if actuator.BulletEnv is not None:
        actuator.setJointSpaceState(np.concatenate([left_joints_value, right_joints_value]).tolist(), env_select="BulletEnv")
        time.sleep(delay)
    else:
        global dummy_jps
        dummy_jps = np.concatenate([left_joints_value, right_joints_value]).tolist()

def getJointSpaceState(actuator):
    if actuator.BulletEnv is not None:
        return actuator.getJointSpaceState()
    else:
        global dummy_jps
        return dummy_jps

def single_workspace_checker(context: GeneralDualArmExecutionCheckingContext, is_left_robot: bool, is_start: bool) -> None:
    """
    Judge whether the pose is in the workspace of a single-arm robot.
    Update the error message in the context if the pose is out of workspace.
    """
    checker_params = context.checker_params
    if is_left_robot: pose = context.pose_start_left if is_start else context.pose_end_left
    else: pose = context.pose_start_right if is_start else context.pose_end_right
    assert pose is not None, "Please set the pose first."
    
    _x = checker_params.x_lim_m[0] < pose.translation[0] < checker_params.x_lim_m[1]
    _y = checker_params.y_lim_m[0] < pose.translation[1] < checker_params.y_lim_m[1]
    _z = checker_params.z_lim_m[0] < pose.translation[2] < checker_params.z_lim_m[1]

    within_workspace = _x and _y and _z

    if is_left_robot:
        context.execution_result.left_arm_within_workspace = within_workspace
    else:
        context.execution_result.right_arm_within_workspace = within_workspace
    
def single_ik_checker(context: GeneralDualArmExecutionCheckingContext, is_left_robot: bool, is_start: bool) -> None:
    """
    Judge whether the pose has at least one IK solution for a single-arm robot to execute
        (we do not consider collision here).
    Update the error message in the context if the pose is not reachable.
    """
    controller = context.controller
    actuator = controller.actuator
    checker_params = context.checker_params
    assert hasattr(checker_params, 'random_ik_ref_pose_num'), "Please set the random IK ref pose number."
    assert hasattr(checker_params, 'random_ik_chosen_ratio'), "Please set the random IK choice ratio."
    random_ik_ref_pose_num = checker_params.random_ik_ref_pose_num
    random_ik_chosen_ratio = checker_params.random_ik_chosen_ratio
    
    jps = getJointSpaceState(actuator)
    jps_left, jps_right = np.split(np.array(jps), 2)

    ref_pose_left_list = list(controller.init_parser.config.motion.robot_l_ref_pos_list) + [
        controller.init_parser.config.fold_one_action.l_home_pose,
        controller.init_parser.config.motion.robot_l_ready,
        controller.init_parser.config.motion.before_fling_pose_l,
        controller.init_parser.config.motion.robot_l_home,
        jps_left,
    ]

    ref_pose_right_list = list(controller.init_parser.config.motion.robot_r_ref_pos_list) + [
        controller.init_parser.config.fold_one_action.r_home_pose,
        controller.init_parser.config.motion.robot_r_ready,
        controller.init_parser.config.motion.before_fling_pose_r,
        controller.init_parser.config.motion.robot_r_home,
        jps_right,
    ]
    
    random_ref_poses_list = [[], []]
    for _ in range(random_ik_ref_pose_num):
        random_ref_pose = controller.getRandomRefInitPose()
        ref_pose_left, ref_pose_right = np.split(np.array(random_ref_pose), 2)
        random_ref_poses_list[0].append(ref_pose_left)
        random_ref_poses_list[1].append(ref_pose_right)
    
    ref_pose_left_list += random_ref_poses_list[0]
    ref_pose_right_list += random_ref_poses_list[1]
    
    robot_type = "left_robot" if is_left_robot else "right_robot"
    if is_left_robot: pose = context.pose_start_left if is_start else context.pose_end_left
    else: pose = context.pose_start_right if is_start else context.pose_end_right
    # (x, y, z, w, rx, ry, rz)
    xyzw = R.from_matrix(pose.rotation).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    tcp_goal_quat = np.concatenate([pose.translation, wxyz])
    
    find_ik = False
    condition_number_candidates = []
    joints_value_candidates = []
    for ref_pose_left, ref_pose_right in zip(ref_pose_left_list, ref_pose_right_list):
        joints_value = controller.singleIKWithQuatFromRef(robot_type, tcp_goal_quat, ref_pose_left, ref_pose_right)
        if robot_type == "left_robot":
            jacobi_eigenvalue = controller.checkJacobiSvd(robot_type, joints_value, jps_right)
        elif robot_type == "right_robot":
            jacobi_eigenvalue = controller.checkJacobiSvd(robot_type, jps_left, joints_value)
        condition_number = 1 / (jacobi_eigenvalue[-1] + 1e-6)
        if len(joints_value) > 1:
            find_ik = True
            condition_number_candidates.append(condition_number)
            joints_value_candidates.append(joints_value)
            if is_left_robot:
                if is_start:
                    context.cached_joints_value_left = context.cached_joints_value_left_start = joints_value
                else:
                    context.cached_joints_value_left = context.cached_joints_value_left_end = joints_value
                if UPDATE_ACTUATOR:
                    setJointSpaceState(actuator, context.cached_joints_value_left, jps_right)
                    setJointSpaceState(actuator, jps_left, jps_right, delay=0)
            else:
                if is_start:
                    context.cached_joints_value_right =  context.cached_joints_value_right_start = joints_value
                else:
                    context.cached_joints_value_right = context.cached_joints_value_right_end = joints_value
                if UPDATE_ACTUATOR:
                    setJointSpaceState(actuator, jps_left, context.cached_joints_value_right)
                    setJointSpaceState(actuator, jps_left, jps_right, delay=0)
    if find_ik:
        chosen_idx = np.argsort(condition_number_candidates)[::-1]\
            [min(int(len(condition_number_candidates) * random_ik_chosen_ratio), len(condition_number_candidates) - 1)]
        if is_left_robot:
            if is_start:
                context.cached_joints_value_left = context.cached_joints_value_left_start = joints_value_candidates[chosen_idx]
            else:
                context.cached_joints_value_left = context.cached_joints_value_left_end = joints_value_candidates[chosen_idx]
        else:
            if is_start:
                context.cached_joints_value_right = context.cached_joints_value_right_start = joints_value_candidates[chosen_idx]
            else:
                context.cached_joints_value_right = context.cached_joints_value_right_end = joints_value_candidates[chosen_idx]
    else:
        if (is_left_robot):
            context.execution_result.left_arm_ik_valid = False
        else:
            context.execution_result.right_arm_ik_valid = False

def dual_joints_value_checker(context: GeneralDualArmExecutionCheckingContext) -> None:
    controller = context.controller
    actuator = controller.actuator
    checker_params = context.checker_params
    jps = getJointSpaceState(actuator)
    jps_left, jps_right = np.split(np.array(jps), 2)
    
    left_joints_value = jps_left if context.cached_joints_value_left == [] else context.cached_joints_value_left
    right_joints_value = jps_right if context.cached_joints_value_right == [] else context.cached_joints_value_right

    if UPDATE_ACTUATOR:
        setJointSpaceState(actuator, left_joints_value, right_joints_value)
        setJointSpaceState(actuator, jps_left, jps_right, delay=0)

    assert hasattr(checker_params, 'jacobian_threshold'), "Please set the jacobian threshold."
    jacobian_threshold = checker_params.jacobian_threshold

    jacobi_eigenvalue_left = controller.checkJacobiSvd("left_robot", left_joints_value, right_joints_value)
    # logger.debug(f'jacobi_eigenvalue_left: {jacobi_eigenvalue_left}， left_joints_value: {left_joints_value}')
    jacobi_eigenvalue_right = controller.checkJacobiSvd("right_robot", left_joints_value, right_joints_value)
    # logger.debug(f'jacobi_eigenvalue_right: {jacobi_eigenvalue_right}, right_joints_value: {right_joints_value}')
    condition_number_left = jacobi_eigenvalue_left[-1]
    condition_number_right = jacobi_eigenvalue_right[-1]
    if condition_number_left < jacobian_threshold:
        logger.warning(
            f'min jacobi_eigenvalue_left {condition_number_left} is lower than threshold {jacobian_threshold}')
        context.execution_result.left_arm_jacobian_valid = False
        return
    if condition_number_right < jacobian_threshold:
        logger.warning(
            f'min jacobi_eigenvalue_right {condition_number_right} is lower than threshold {jacobian_threshold}')
        context.execution_result.right_arm_jacobian_valid = False
        return

    assert hasattr(checker_params, 'joints_value_limit'), "Please set the joints value limit."
    joints_value_limit = dict(checker_params.joints_value_limit)
    for joint_idx, limit_list in joints_value_limit.items():
        within_limit_left = False
        within_limit_right = False
        joint_idx = int(joint_idx)
        for limit_deg in limit_list:
            limit = [math.radians(limit_deg[i]) for i in range(len(limit_deg))]
            if limit[0] < left_joints_value[joint_idx] < limit[1]:
                within_limit_left = True
            if limit[0] < right_joints_value[joint_idx] < limit[1]:
                within_limit_right = True
        if not within_limit_left:
            logger.debug(f"Joint {joint_idx} value {math.degrees(left_joints_value[joint_idx])} is out of limit {limit_list}.")
            context.execution_result.left_arm_joints_value_valid = False
            break
        if not within_limit_right:
            logger.debug(f"Joint {joint_idx} value {math.degrees(right_joints_value[joint_idx])} is out of limit {limit_list}.")
            context.execution_result.right_arm_joints_value_valid = False
            break

def dual_collision_checker(context: GeneralDualArmExecutionCheckingContext, type: list) -> None:
    """
    Judge whether the poses are collision-free for a dual-arm robot to execute.
    Update the error message in the context if the pose is not collision-free.
    """
    controller = context.controller
    actuator = controller.actuator
    checker_params = context.checker_params
    jps = getJointSpaceState(actuator)
    jps_left, jps_right = np.split(np.array(jps), 2)
    
    left_joints_value = jps_left if context.cached_joints_value_left == [] else context.cached_joints_value_left
    right_joints_value = jps_right if context.cached_joints_value_right == [] else context.cached_joints_value_right
    
    if 'general' in type:
        no_collision = context.controller.checkCollisionState(left_joints_value, right_joints_value)
        if not no_collision:
            logger.debug("General collision detected.")
        context.execution_result.no_collision &= no_collision
    
    if context.execution_result.no_collision and 'between_robots_distance' in type:
        assert hasattr(checker_params, 'between_robots_collision_distance_limit'), "Please set the collision distance limit."
        distance_limit = checker_params.between_robots_collision_distance_limit
        min_distance = context.controller.getDualRobotMinDistance(left_joints_value, right_joints_value)
        no_collision = min_distance > distance_limit
        if not no_collision:
            logger.debug(f"Collision between robots detected, min distance: {min_distance}.")
        context.execution_result.no_collision &= no_collision
    
    if context.execution_result.no_collision and 'robots_desktop_distance' in type:
        assert hasattr(checker_params, 'robots_desktop_collision_distance_limit'), "Please set the collision distance limit."
        distance_limit = checker_params.robots_desktop_collision_distance_limit
        min_distance = context.controller.getRobotDesktopMinDistance(left_joints_value, right_joints_value, output=False)
        no_collision = min_distance > distance_limit
        if not no_collision:
            logger.debug(f"Collision between robots and desktop detected, min distance: {min_distance}.")
        context.execution_result.no_collision &= no_collision
        
def dual_ik_joints_value_collision_checker(context: GeneralDualArmExecutionCheckingContext, is_start: bool, type:list) -> None:
    # cache context for revert
    context.cache()
    controller = context.controller
    actuator = controller.actuator
    checker_params = context.checker_params
    assert hasattr(checker_params, 'random_ik_ref_pose_num'), "Please set the random IK ref pose number."
    assert hasattr(checker_params, 'random_ik_chosen_ratio'), "Please set the random IK choice ratio."
    random_ik_ref_pose_num = checker_params.random_ik_ref_pose_num
    random_ik_chosen_ratio = checker_params.random_ik_chosen_ratio
    
    jps = getJointSpaceState(actuator)
    jps_left, jps_right = np.split(np.array(jps), 2)

    logger.debug(
        f'controller.init_parser.config.fold_one_action.l_home_pose: {controller.init_parser.config.fold_one_action.l_home_pose}')
    ref_pose_left_list = list(controller.init_parser.config.motion.robot_l_ref_pos_list) + [
        controller.init_parser.config.fold_one_action.l_home_pose,
        controller.init_parser.config.motion.robot_l_ready,
        controller.init_parser.config.motion.before_fling_pose_l,
        controller.init_parser.config.motion.robot_l_home,
        jps_left,
    ]

    ref_pose_right_list = list(controller.init_parser.config.motion.robot_r_ref_pos_list) + [
        controller.init_parser.config.fold_one_action.r_home_pose,
        controller.init_parser.config.motion.robot_r_ready,
        controller.init_parser.config.motion.before_fling_pose_r,
        controller.init_parser.config.motion.robot_r_home,
        jps_right,
    ]
    
    random_ref_poses_list = [[], []]
    for _ in range(random_ik_ref_pose_num):
        random_ref_pose = controller.getRandomRefInitPose()
        ref_pose_left, ref_pose_right = np.split(np.array(random_ref_pose), 2)
        random_ref_poses_list[0].append(ref_pose_left)
        random_ref_poses_list[1].append(ref_pose_right)
    
    ref_pose_left_list += random_ref_poses_list[0]
    ref_pose_right_list += random_ref_poses_list[1]
    
    ref_poses_pair_list = list(itertools.product(ref_pose_left_list, ref_pose_right_list))
    
    pose_left = context.pose_start_left if is_start else context.pose_end_left
    pose_right = context.pose_start_right if is_start else context.pose_end_right
    # (x, y, z, w, rx, ry, rz)
    xyzw = R.from_matrix(pose_left.rotation).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    tcp_goal_quat_left = np.concatenate([pose_left.translation, wxyz])
    xyzw = R.from_matrix(pose_right.rotation).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    tcp_goal_quat_right = np.concatenate([pose_right.translation, wxyz])
    
    find_ik = False
    condition_number_candidates = []
    joints_value_left_candidates = []
    joints_value_right_candidates = []
    for ref_pose_left, ref_pose_right in ref_poses_pair_list:
        joints_value_left = controller.singleIKWithQuatFromRef("left_robot", tcp_goal_quat_left, ref_pose_left, ref_pose_right)
        joints_value_right = controller.singleIKWithQuatFromRef("right_robot", tcp_goal_quat_right, ref_pose_left, ref_pose_right)
        jacobi_eigenvalue_left = controller.checkJacobiSvd("left_robot", joints_value_left, joints_value_right)
        jacobi_eigenvalue_right = controller.checkJacobiSvd("right_robot", joints_value_left, joints_value_right)
        condition_number_left = 1 / (jacobi_eigenvalue_left[-1] + 1e-6)
        condition_number_right = 1 / (jacobi_eigenvalue_right[-1] + 1e-6)
        condition_number = max(condition_number_left, condition_number_right)
        if len(joints_value_left) > 1 and len(joints_value_right) > 1:
            if is_start:
                context.cached_joints_value_left = context.cached_joints_value_left_start = joints_value_left
                context.cached_joints_value_right = context.cached_joints_value_right_start = joints_value_right
            else:
                context.cached_joints_value_left = context.cached_joints_value_left_end = joints_value_left
                context.cached_joints_value_right = context.cached_joints_value_right_end = joints_value_right
            if UPDATE_ACTUATOR:
                setJointSpaceState(actuator, context.cached_joints_value_left, context.cached_joints_value_right)
                setJointSpaceState(actuator, jps_left, jps_right, delay=0)
            
            dual_joints_value_checker(context)
            if not context.execution_result.overall_success:
                context.revert()
                continue
            dual_collision_checker(context, type)
            if not context.execution_result.overall_success:
                context.revert()
                continue
                
            find_ik = True
            condition_number_candidates.append(condition_number)
            joints_value_left_candidates.append(joints_value_left)
            joints_value_right_candidates.append(joints_value_right)
    
    if find_ik:
        chosen_idx = np.argsort(condition_number_candidates)[::-1]\
            [min(int(len(condition_number_candidates) * random_ik_chosen_ratio), len(condition_number_candidates) - 1)]
        if is_start:
            context.cached_joints_value_left = context.cached_joints_value_left_start = joints_value_left_candidates[chosen_idx]
            context.cached_joints_value_right = context.cached_joints_value_right_start = joints_value_right_candidates[chosen_idx]
        else:
            context.cached_joints_value_left = context.cached_joints_value_left_end = joints_value_left_candidates[chosen_idx]
            context.cached_joints_value_right = context.cached_joints_value_right_end = joints_value_right_candidates[chosen_idx]
    else:
        context.execution_result.dual_arm_ik_valid = False

def dual_planning_checker(context: GeneralDualArmExecutionCheckingContext) -> None:
    """
    Judge whether the p
    Update the error message in the context if the pose is not plannable.
    Add the planning result to the context for further checking.
    """
    assert context.cached_joints_value_left_end != [] or context.cached_joints_value_right_end != [], "Please run the IK checker first."
    controller = context.controller
    actuator = controller.actuator
    jps = getJointSpaceState(actuator)
    jps_left, jps_right = np.split(np.array(jps), 2)
    
    left_joints_value_start = jps_left if context.cached_joints_value_left_start == [] else context.cached_joints_value_left_start
    right_joints_value_start = jps_right if context.cached_joints_value_right_start == [] else context.cached_joints_value_right_start
    left_joints_value_end = jps_left if context.cached_joints_value_left_end == [] else context.cached_joints_value_left_end
    right_joints_value_end = jps_right if context.cached_joints_value_right_end == [] else context.cached_joints_value_right_end
    
    start_point = np.concatenate([left_joints_value_start, right_joints_value_start]).tolist()
    waypoints = controller.runDualRobotJPoseForPlan(start_point, left_joints_value_end, right_joints_value_end)
    if len(waypoints) > 0:
        context.cached_waypoints = waypoints
        if UPDATE_ACTUATOR:
            actuator.runJointSpaceTrajs(waypoints)
            setJointSpaceState(actuator, jps_left, jps_right, delay=0)
    else:
        context.execution_result.dual_arm_planning_valid = False

def dual_trajectory_checker(context: GeneralDualArmExecutionCheckingContext, type: list) -> None:
    """
    Judge whether the waypoints is valid for a single-arm robot to execute.
    Update the error message in the context if the waypoints are not valild.
    """
    assert context.cached_waypoints != [], "Please run the planning checker first."
    
    controller = context.controller
    checker_params = context.checker_params
    cached_waypoints = context.cached_waypoints
    
    if 'joints' in type:
        assert hasattr(checker_params, 'trajectory_critical_joints') and hasattr(checker_params, 'trajectory_angle_limits'), \
            "Please set the joints trajectory distance limits."
        
        left_waypoints = np.array([waypoint.waypoint[:7] for waypoint in cached_waypoints])
        right_waypoints = np.array([waypoint.waypoint[7:] for waypoint in cached_waypoints])
        
        trajectory_critical_joints = checker_params.trajectory_critical_joints
        trajectory_angle_limits = checker_params.trajectory_angle_limits
        
        if np.max(np.abs(np.max(left_waypoints[:, trajectory_critical_joints], axis=0) - np.min(left_waypoints[:, trajectory_critical_joints], axis=0))) \
                > trajectory_angle_limits:
            context.execution_result.left_arm_trajectory_valid = False
        if np.max(np.abs(np.max(right_waypoints[:, trajectory_critical_joints], axis=0) - np.min(right_waypoints[:, trajectory_critical_joints], axis=0))) \
                > trajectory_angle_limits:
            context.execution_result.right_arm_trajectory_valid = False
    
    if 'ends' in type:
        assert hasattr(checker_params, 'trajectory_distance_limits') and hasattr(checker_params, 'trajectory_distance_start_threshold') \
            and hasattr(checker_params, 'trajectory_distance_limits_delta'), "Please set the trajectory distance limits." \
            "Please set the ends trajectory distance limits."
        
        c_waypoints = np.array(controller.calCartesianPath(waypoints=cached_waypoints))
        c_left_waypoints = c_waypoints[:, 0]
        c_right_waypoints = c_waypoints[:, 1]
        
        initial_distance = np.linalg.norm(c_left_waypoints[0, :3] - c_right_waypoints[0, :3])
        distance_limits = checker_params.trajectory_distance_limits
        distance_start_threshold = checker_params.trajectory_distance_start_threshold
        distance_limits_delta = checker_params.trajectory_distance_limits_delta
        if np.linalg.norm(c_left_waypoints[0, :3] - c_right_waypoints[0, :3]) < distance_start_threshold:
            distance_limits += distance_limits_delta
        if np.max(np.linalg.norm(c_left_waypoints[:, :3] - c_right_waypoints[:, :3], axis=1)) > initial_distance + distance_limits:
            context.execution_result.left_arm_trajectory_valid = False
            context.execution_result.right_arm_trajectory_valid = False