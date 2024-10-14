import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional

class FPS:
    "Farthest Point Sampling"
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)

        self.grouping_radius = None
        self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()
        self.labels = None

        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.n_selected_pts = 1

    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts

    def step(self):
        if self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
            dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[res_selected_idx]

            self.n_selected_pts += 1
        else:
            print("Got enough number samples")


    def fit(self):
        for _ in range(1, self.n_samples):
            self.step()
        return self.get_selected_pts()

    def group(self, radius):
        self.grouping_radius = radius   # the grouping radius is not actually used
        dists = self.dist_pts_to_selected

        # Ignore the "points"-"selected" relations if it's larger than the radius
        dists = np.where(dists > radius, dists+1000000*radius, dists)

        # Find the relation with the smallest distance.
        # NOTE: the smallest distance may still larger than the radius.
        self.labels = np.argmin(dists, axis=1)
        return self.labels


    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)

def pick_n_points_from_pcd(point_cloud: o3d.geometry.PointCloud, n: int) -> Tuple[
    List[np.ndarray], List[np.ndarray], Optional[Exception]]:
    """pick 2 points from point cloud, with interaction

    Args:
        point_cloud (o3d.geometry.PointCloud): point cloud
        n (int): number of points

    Returns:
         Tuple[List[np.ndarray], List[np.ndarray], Optional[Exception]]: points xyz, points index, exception if any
    """
    # coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # point_cloud += coordinate.pcd
    if n <= 0:
        return [], [], None
    vis1 = o3d.visualization.VisualizerWithEditing()
    window_name = f'Please press Shift and click {n} Points: Left Right. Press Q to exit.'
    vis1.create_window(window_name, width=2560, height=1440)
    vis1.add_geometry(point_cloud)

    view_control = vis1.get_view_control()
    view_control.set_zoom(0.1)

    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    if len(pts_sel) != n:
        return [], [], Exception(f"Please select {n} points")

    pcd_points = np.asarray(point_cloud.points)
    pts = pcd_points[pts_sel]
    return pts, pts_sel, None

def random_downsample_point_cloud(point_cloud: np.ndarray, num_points: int) -> np.ndarray:
    """
    Randomly downsample a point cloud to a specified number of points.

    Parameters:
    point_cloud (numpy.ndarray): The original point cloud, shape (N, D) where N is the number of points and D is the dimension.
    num_points (int): The desired number of points after downsampling.

    Returns:
    numpy.ndarray: The downsampled point cloud, shape (num_points, D).
    """
    # Check if the number of points to downsample is greater than the original number of points
    if num_points > point_cloud.shape[0]:
        raise ValueError("The number of points to downsample must be less than the original number of points.")

    # Generate random indices to select from the original point cloud
    indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)

    # Select the points at the generated indices
    downsampled_point_cloud = point_cloud[indices, :]

    return downsampled_point_cloud