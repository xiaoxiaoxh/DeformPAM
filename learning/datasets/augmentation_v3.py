import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class SequentialV3:
    def __init__(self, augmentors, use_torch=False):
        self.augmentors = augmentors
        self.use_torch = use_torch

    def __call__(self, points, poses, dummy_keypoint_num=None):
        if self.use_torch:
            points = torch.from_numpy(points)
            poses = torch.from_numpy(poses)
        for a in self.augmentors:
            if isinstance(a, AffineV3):
                points, poses = a(points, poses, self.use_torch, dummy_keypoint_num)
            else:
                points, poses = a(points, poses, self.use_torch)
        if self.use_torch:
            points = points.numpy()
            poses = poses.numpy()
        return points, poses


class AffineV3:
    def __init__(self,
                 x_trans_range=None,
                 y_trans_range=None,
                 rot_angle_range=None,
                 scale_range=None,
                 trans_place_pose=False,
                 use_zero_center=False):
        self.x_trans_range = x_trans_range
        self.y_trans_range = y_trans_range
        self.rot_angle_range = rot_angle_range
        self.scale_range = scale_range
        self.trans_place_pose = trans_place_pose  # whether to transform place point in frame 2
        self.use_zero_center=use_zero_center

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    def __call__(self, points, poses, use_torch=False, dummy_keypoint_num=None):
        if use_torch:
            # hack for keypoints
            if dummy_keypoint_num is not None:
                center = ((torch.max(points[:-dummy_keypoint_num], dim=0)[0] + torch.min(points[:-dummy_keypoint_num], dim=0)[0]) / 2)[None, :]
            else:
                center = ((torch.max(points, dim=0)[0] + torch.min(points, dim=0)[0]) / 2)[None, :]
        else:
            # hack for keypoints
            if dummy_keypoint_num is not None:
                center = ((np.max(points[:-dummy_keypoint_num], axis=0) + np.min(points[:-dummy_keypoint_num], axis=0)) / 2)[None, :]
            else:
                center = ((np.max(points, axis=0) + np.min(points, axis=0)) / 2)[None, :]

        rot_angle = self.rand_uniform(low=self.rot_angle_range[0], high=self.rot_angle_range[1]) if self.rot_angle_range else 0.0
        x_trans = self.rand_uniform(low=self.x_trans_range[0], high=self.x_trans_range[1]) if self.x_trans_range else 0.0
        y_trans = self.rand_uniform(low=self.y_trans_range[0], high=self.y_trans_range[1]) if self.y_trans_range else 0.0
        scale_trans = self.rand_uniform(low=self.scale_range[0], high=self.scale_range[1]) if self.scale_range else 1.0
        offset_trans = np.array([[x_trans, y_trans, 0.]]).astype(np.float32)
        if use_torch:
            offset_trans = torch.from_numpy(offset_trans)

        rot_mat = R.from_euler(
            'z', rot_angle, degrees=False
        ).as_matrix().astype(np.float32)
        if use_torch:
            rot_mat = torch.from_numpy(rot_mat)

        if self.use_zero_center:
            points = ((points - center) * scale_trans) @ rot_mat.T + offset_trans
            # no zero-center for z axis
            points[:, 2] += center[:, 2]
        else:
            points = ((points - center) * scale_trans) @ rot_mat.T + center + offset_trans
        for idx in (0, 1, 2, 3) if self.trans_place_pose else (0, 2):
            if self.use_zero_center:
                poses[idx, :3] = ((poses[idx, :3][None, :] - center) * scale_trans) @ rot_mat.T + offset_trans
                # no zero-center for z axis
                poses[idx, 2] += center[0, 2]
            else:
                poses[idx, :3] = ((poses[idx, :3][None, :] - center) * scale_trans) @ rot_mat.T + center + offset_trans 
            poses[idx, -1] = poses[idx, -1] - rot_angle / 180.0 * np.pi

        # clip angle range in [-pi, pi]
        idxs = poses[:, -1] > np.pi
        poses[idxs, -1] = poses[idxs, -1] - 2 * np.pi
        idxs = poses[:, -1] < -np.pi
        poses[idxs, -1] = poses[idxs, -1] + 2 * np.pi
        return points, poses


class AutoPermutePoseV3:
    def __init__(self):
        pass

    def __call__(self, points, poses, use_torch=False):
        if poses[0, 0] > poses[2, 0]:  # x1 > x2, frame 1
            # permute lef-right pose based on x-coordinate
            if use_torch:
                poses[0, :], poses[2, :] = poses[2, :].clone(), poses[0, :].clone()
            else:
                poses[0, :], poses[2, :] = poses[2, :].copy(), poses[0, :].copy()
        if poses[1, 0] > poses[3, 0]:  # x1 > x2, frame 2
            # permute lef-right pose based on x-coordinate
            if use_torch:
                poses[1, :], poses[3, :] = poses[3, :].clone(), poses[1, :].clone()
            else:
                poses[1, :], poses[3, :] = poses[3, :].copy(), poses[1, :].copy()
        return points, poses


class RandomPermutePoseV3:
    def __init__(self):
        pass

    def __call__(self, points, poses, use_torch=False):
        p1 = torch.rand(1)
        if p1 < 0.5:  # frame 1
            # permute lef-right pick pose
            if use_torch:
                poses[0, :], poses[2, :] = poses[2, :].clone(), poses[0, :].clone()
            else:
                poses[0, :], poses[2, :] = poses[2, :].copy(), poses[0, :].copy()
        return points, poses


class FlipV3:
    def __init__(self, lr_percent=None, ud_percent=None, trans_place_pose=False):
        self.lr_percent = lr_percent
        self.ud_percent = ud_percent
        self.trans_place_pose = trans_place_pose

    def __call__(self, points, poses, use_torch=False):
        if use_torch:
            center = ((torch.max(points, dim=0)[0] + torch.min(points, dim=0)[0]) / 2)[None, :]
        else:
            center = ((np.max(points, axis=0) + np.min(points, axis=0)) / 2)[None, :]
        p_ud, p_lr = torch.rand(2)

        if p_ud < self.ud_percent:  # UD
            points[:, 1] = 2 * center[:, 1] - points[:, 1]  # y-axis
            poses[:, 1] = 2 * center[:, 1] - poses[:, 1]
            for idx in (0, 1, 2, 3) if self.trans_place_pose else (0, 2):
                # only flip frame 1 (grasp point), nor frame 2 (place point)
                poses[idx, -1] = -poses[idx, -1]  # change theta

        if p_lr < self.lr_percent:  # LR
            points[:, 0] = 2 * center[:, 0] - points[:, 0]  # x-axis
            poses[:, 0] = 2 * center[:, 0] - poses[:, 0]  # x-axis
            for idx in (0, 1, 2, 3) if self.trans_place_pose else (0, 2):
                # only flip frame 1 (grasp point), nor frame 2 (place point)
                poses[idx, -1] = np.pi - poses[idx, -1]  # change theta

        # clip angle range in [-pi, pi]
        idxs = poses[:, -1] > np.pi
        poses[idxs, -1] = poses[idxs, -1] - 2 * np.pi
        idxs = poses[:, -1] < -np.pi
        poses[idxs, -1] = poses[idxs, -1] + 2 * np.pi
        return points, poses


class DepthV3:
    def __init__(self, scale_range=(0.2, 1.2), trans_range=(0.0, 0.0)):
        self.scale_range = scale_range
        self.trans_range = trans_range

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    def __call__(self, points, poses, use_torch=False):
        scale = self.rand_uniform(self.scale_range[0], self.scale_range[1])
        # TODO: support points with non-zero plane height
        points[:, 2] = points[:, 2] * scale  # z-axis
        poses[:, 2] = poses[:, 2] * scale
        offset = self.rand_uniform(self.trans_range[0], self.trans_range[1])
        points[:, 2] = points[:, 2] + offset
        poses[:, 2] = poses[:, 2] + offset
        return points, poses
