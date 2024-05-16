import typing
import pathlib
import numpy as np

from torch.utils.data import Dataset

Transform = typing.Callable[[np.ndarray], np.ndarray]


class SemSegDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        point_transforms: tuple[Transform] = tuple(),
        label_transforms: tuple[Transform] = tuple(),
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.point_transforms = point_transforms
        self.label_transforms = label_transforms
        self.files = tuple(self.data_dir.glob("*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.files[idx])
        points, labels = data[:, :-1], data[:, -1]
        for transform in self.point_transforms:
            points = transform(points)
        for transform in self.label_transforms:
            labels = transform(labels)
        return points, labels
