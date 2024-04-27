from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from piecharts.data.dataset import PiechartDataset
from piecharts.nn.models.smp_models import PSMModel
from piecharts.training.train import train

BATCH_SIZE = 16
def main():
    model = PSMModel(True)
    train_dataset = PiechartDataset(Path("data") / "raw", "train", "train", (256, 256))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = PiechartDataset(Path("data") / "raw", "train", "val", (256, 256))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter(log_dir="./tensorboard/")
    train(model, train_dataloader, val_dataloader, writer, 100)


if __name__ == "__main__":
    main()
