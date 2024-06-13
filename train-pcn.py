import time
import typing
import argparse
import os
import datetime
import dataclasses
import json

import torch
import torch.utils.data.dataloader
import numpy as np
import tqdm

import pcs.models.pointconv_simple
import pcs.dataset
import pcs.preprocess

NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./out/checkpoints/"
HISTORY_DIR = "./out/history/"
SAVE_EVERY = 5

PC_TRANSFORMS = (
    pcs.preprocess.normalize_position,
    pcs.preprocess.normalize_colors,
    pcs.preprocess.normalize_intensity,
    lambda x: torch.tensor(x.T, dtype=torch.float32),
)

LABEL_TRANSFORMS = (lambda x: torch.tensor(x - 1, dtype=torch.long),)


class Args(typing.NamedTuple):
    data_dir: str
    batch_size: int
    max_epochs: int
    model_name: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pointconv",
        help="Name of the model to train",
    )
    return parser.parse_args()


@dataclasses.dataclass
class Dataloaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


@dataclasses.dataclass
class History:
    model_name: str
    batch_size: int
    epochs: int
    dataset: str
    epoch_times: list[float] = dataclasses.field(default_factory=list)
    train_loss: list[float] = dataclasses.field(default_factory=list)
    val_loss: list[float] = dataclasses.field(default_factory=list)
    test_acc: list[float] = dataclasses.field(default_factory=list)
    test_iou: list[float] = dataclasses.field(default_factory=list)
    start_time: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    def __post_init__(self):
        if self.model_name == "pointconv":
            self.model_name += f"_{self.start_time}"

    def export(self):
        filename = f"{HISTORY_DIR}{self.model_name}_hist.json"
        with open(filename, "w") as f:
            json.dump(dataclasses.asdict(self), f)


def calculate_per_class_stats(
    model: torch.nn.Module, dataset: torch.utils.data.DataLoader, n_classes: int
) -> tuple[list[float], list[float]]:
    all_labels = []
    all_predictions = []
    for points, labels in tqdm.tqdm(dataset):
        with torch.no_grad():
            predictions = model(points)
        all_labels.append(labels)
        all_predictions.append(predictions.argmax(dim=1))
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    per_class_accuracy = []
    for i in range(n_classes):
        mask = all_labels == i
        correct = (all_predictions[mask] == i).sum().item()
        total = mask.sum().item()
        try:
            per_class_accuracy.append(correct / total)
        except ZeroDivisionError:
            per_class_accuracy.append(-1)
    per_class_iou = []
    for i in range(n_classes):
        mask = all_labels == i
        intersection = (all_predictions[mask] == i).sum().item()
        union = mask.sum().item() + (all_predictions == i).sum().item() - intersection
        try:
            per_class_iou.append(intersection / union)
        except ZeroDivisionError:
            per_class_iou.append(-1)
    return per_class_accuracy, per_class_iou


def load_data(data_dir: str, batch_size: int) -> Dataloaders:
    datasets = {
        split: pcs.dataset.SemSegDataset(
            data_dir=data_dir + split,
            point_transforms=PC_TRANSFORMS,
            label_transforms=LABEL_TRANSFORMS,
            load_device=DEVICE,
        )
        for split in ["train", "val", "test"]
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train")
        )
        for split, dataset in datasets.items()
    }
    return Dataloaders(**loaders)


def main():
    args = parse_args()
    dataloaders = load_data(args.data_dir, args.batch_size)
    epoch_length = len(dataloaders.train)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    print(f"Training PointConvNet using device:{DEVICE}")
    print(f"Loaded {len(dataloaders.train)} training samples")

    model = pcs.models.pointconv_simple.PointConvNet(
        features=4, classes=NUM_CLASSES
    ).to(DEVICE)
    loss_fn = torch.nn.NLLLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = History(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.batch_size,
        dataset=args.data_dir.replace("\\", "/").split("/")[-2],
    )
    for epoch in range(args.max_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        pbar = tqdm.tqdm(dataloaders.train, total=epoch_length)
        train_loss = []
        for points, labels in pbar:
            optimizer.zero_grad()
            pred = model(points)
            loss = loss_fn(pred, labels)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(
                f"Epoch {epoch + 1}/{args.max_epochs}, loss: {loss.item()}"
            )
        pbar.close()
        history.train_loss.append(np.mean(train_loss))
        model.eval()
        with torch.no_grad():
            losses = []
            for points, labels in dataloaders.val:
                pred = model(points)
                loss = loss_fn(pred, labels)
                losses.append(loss.item())
            history.val_loss.append(np.mean(losses))
            print(f"Validation loss: {np.mean(losses)}")

        if epoch and (epoch % SAVE_EVERY == 0):
            torch.save(model.state_dict(), SAVE_DIR + history.model_name + f"_c{epoch}.pt")
            print(f"Saved model at epoch {epoch}")

        epoch_time = time.perf_counter() - epoch_start_time
        history.epoch_times.append(epoch_time)
        history.export()

    torch.save(model.state_dict(), SAVE_DIR + history.model_name + ".pt")
    print("Training complete")

    model.eval()
    with torch.no_grad():
        losses = []
        for points, labels in dataloaders.test:
            pred = model(points)
            loss = loss_fn(pred, labels)
            losses.append(loss.item())
        print(f"Test loss: {np.mean(losses)}")

    acc, iou = calculate_per_class_stats(model, dataloaders.test, NUM_CLASSES)
    history.test_acc = acc
    history.test_iou = iou
    history.export()


if __name__ == "__main__":
    main()
