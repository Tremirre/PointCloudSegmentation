import typing
import pathlib
import numpy as np

from torch import device
from torch.utils.data import Dataset

Transform = typing.Callable[[np.ndarray], np.ndarray]


class SemSegDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        point_transforms: tuple[Transform] = tuple(),
        label_transforms: tuple[Transform] = tuple(),
        load_device: device = device("cpu"),
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.point_transforms = point_transforms
        self.label_transforms = label_transforms
        self.files = tuple(self.data_dir.glob("*.npy"))
        self.load_device = load_device
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx not in self.cache:
            data = np.load(self.files[idx])
            points, labels = data[:, :-1], data[:, -1]
            for transform in self.point_transforms:
                points = transform(points)
            for transform in self.label_transforms:
                labels = transform(labels)
            self.cache[idx] = points, labels
        else:
            points, labels = self.cache[idx]
        return points.to(self.load_device), labels.to(self.load_device)
