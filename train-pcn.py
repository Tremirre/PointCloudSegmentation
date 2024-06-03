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

BATCH_SIZE = 8
NUM_CLASSES = 8
DATA_DIR = "./data/aggregated/bild/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 10
SAVE_DIR = "./out/checkpoints/"
HISTORY_DIR = "./out/history/"
SAVE_EVERY = 10


@dataclasses.dataclass
class Dataloaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


@dataclasses.dataclass
class History:
    train_loss: list[list[float]] = dataclasses.field(default_factory=list)
    val_loss: list[float] = dataclasses.field(default_factory=list)


def load_data(data_dir: str, batch_size: int) -> Dataloaders:
    p_transforms = lambda x: torch.tensor(x.T, dtype=torch.float32).to(DEVICE)
    l_transforms = lambda x: torch.tensor(x - 1, dtype=torch.long).to(DEVICE)
    datasets = {
        split: pcs.dataset.SemSegDataset(
            data_dir=data_dir + split,
            point_transforms=(p_transforms,),
            label_transforms=(l_transforms,),
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
    dataloaders = load_data(DATA_DIR, BATCH_SIZE)
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

    history = History()
    for epoch in range(MAX_EPOCHS):
        model.train()
        pbar = tqdm.tqdm(dataloaders.train, total=epoch_length)
        for i, (points, labels) in enumerate(pbar):
            optimizer.zero_grad()
            pred = model(points)
            loss = loss_fn(pred, labels)
            loss.backward()
            history.train_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(f"Epoch {epoch + 1}/{MAX_EPOCHS}, loss: {loss.item()}")
        pbar.close()

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
            torch.save(model.state_dict(), SAVE_DIR + f"model_{epoch}.pt")
            print(f"Saved model at epoch {epoch}")

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), SAVE_DIR + f"model_final_{now}.pt")
    print("Training complete")

    model.eval()
    with torch.no_grad():
        losses = []
        for points, labels in dataloaders.test:
            pred = model(points)
            loss = loss_fn(pred, labels)
            losses.append(loss.item())
        print(f"Test loss: {np.mean(losses)}")

    history_file = HISTORY_DIR + f"history_{now}.json"
    with open(history_file, "w") as f:
        json.dump(dataclasses.asdict(history), f)
    print(f"Saved history to {history_file}")


if __name__ == "__main__":
    main()
