import numpy as np
from scipy.spatial.transform import Rotation


def unity_hand_pose_to_open3d(left_hand_positions: np.ndarray, right_hand_positions: np.ndarray,
                              left_hand_euler_angles: np.ndarray, right_hand_euler_angles: np.ndarray,
                              mode: str = 'simple_hand') -> tuple:
    """
    change from Unity coornidates to Open3D coordinates
    change euler angles to quaternions
    """
    if mode == 'simple_hand':
        bones_num = left_hand_positions.shape[0]
        if bones_num > 20:
            left_thumb_position = left_hand_positions[20, :]
            left_index_position = left_hand_positions[21, :]
            right_thumb_position = right_hand_positions[20, :]
            right_index_position = right_hand_positions[21, :]

            left_thumb_euler_angles = left_hand_euler_angles[20, :]
            left_index_euler_angles = left_hand_euler_angles[21, :]
            right_thumb_euler_angles = right_hand_euler_angles[20, :]
            right_index_euler_angles = right_hand_euler_angles[21, :]
        elif bones_num == 2:
            left_thumb_position = left_hand_positions[0, :]
            left_index_position = left_hand_positions[1, :]
            right_thumb_position = right_hand_positions[0, :]
            right_index_position = right_hand_positions[1, :]

            left_thumb_euler_angles = left_hand_euler_angles[0, :]
            left_index_euler_angles = left_hand_euler_angles[1, :]
            right_thumb_euler_angles = right_hand_euler_angles[0, :]
            right_index_euler_angles = right_hand_euler_angles[1, :]
        else:
            raise NotImplementedError

        # convert Unity coordinates to Open3D coordinates
        left_thumb_position[2] = -left_thumb_position[2]
        left_index_position[2] = -left_index_position[2]
        right_thumb_position[2] = -right_thumb_position[2]
        right_index_position[2] = -right_index_position[2]
        hand_positions = np.stack([left_thumb_position, left_index_position,
                                   right_thumb_position, right_index_position], axis=0)

        # TODO: check whether is correct
        hand_euler_angles = np.stack([left_thumb_euler_angles, left_index_euler_angles,
                                      right_thumb_euler_angles, right_index_euler_angles], axis=0)
        hand_euler_angles[:, 2] = -hand_euler_angles[:, 2]
        quaternion_list = []
        for idx in range(4):
            r = Rotation.from_euler('zxy', hand_euler_angles[idx, :], degrees=True)
            quat = r.as_quat()
            quaternion_list.append(quat)
        hand_quats = np.stack(quaternion_list, axis=0)
        return hand_positions, hand_quats
    elif mode == 'full_hand':
        hand_positions = np.concatenate([left_hand_positions, right_hand_positions], axis=0)
        hand_positions[:, 2] = -hand_positions[:, 2]
        # TODO: check whether is correct
        hand_euler_angles = np.concatenate([left_hand_euler_angles, right_hand_euler_angles], axis=0)
        hand_euler_angles[:, 2] = -hand_euler_angles[:, 2]
        quaternion_list = []
        for idx in range(hand_euler_angles.shape[0]):
            r = Rotation.from_euler('zxy', hand_euler_angles[idx, :], degrees=True)
            quat = r.as_quat()
            quaternion_list.append(quat)
        hand_quats = np.stack(quaternion_list, axis=0)
        return hand_positions, hand_quats
    else:
        raise NotImplementedError


def open3d_hand_pose_to_unity(hand_positions: np.ndarray, hand_quats: np.ndarray, mode: str = 'simple_hand') -> tuple:
    """
    change from Open3D coornidates to Unity coordinates
    change quaternions to euler angles
    """
    if mode == 'simple_hand':
        left_thumb_position, left_index_position, right_thumb_position, right_index_position = \
            hand_positions[0, :], hand_positions[1, :], hand_positions[2, :], hand_positions[3, :]
        # convert Open3D coordinates to Unity coordinates
        left_thumb_position[2] = -left_thumb_position[2]
        left_index_position[2] = -left_index_position[2]
        right_thumb_position[2] = -right_thumb_position[2]
        right_index_position[2] = -right_index_position[2]

        hand_euler_angle_list = []
        for idx in range(hand_quats.shape[0]):
            r = Rotation.from_quat(hand_quats[idx, :])
            euler_angles = r.as_euler('zxy', degrees=True)
            # convert Open3D coordinates to Unity coordinates
            euler_angles[2] = -euler_angles[2]
            hand_euler_angle_list.append(euler_angles)
        left_thumb_euler_angles, left_index_euler_angles, \
            right_thumb_euler_angles, right_index_euler_angles = hand_euler_angle_list
        return (left_thumb_position, left_index_position, right_thumb_position, right_index_position), \
               (left_thumb_euler_angles, left_index_euler_angles,
                right_thumb_euler_angles, right_index_euler_angles)
    else:
        raise NotImplementedError