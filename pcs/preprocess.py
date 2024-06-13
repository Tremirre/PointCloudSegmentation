import numpy as np


def normalize_position(point_cloud: np.ndarray) -> np.ndarray:
    centroid = np.mean(point_cloud[:, :3], axis=1)
    point_cloud[:, :3] -= centroid[:, None]
    return point_cloud


def normalize_colors(point_cloud: np.ndarray) -> np.ndarray:
    point_cloud[:, 4:] /= 255
    point_cloud[:, 4:] -= 0.5
    return point_cloud


def normalize_intensity(point_cloud: np.ndarray) -> np.ndarray:
    point_cloud[:, 3] /= 2048
    return point_cloud
