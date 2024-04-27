import io
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torchmetrics.classification
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    model: nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, writer: SummaryWriter, epochs: int, model_dir: Path, lr: float, loss: str
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss == "focal":
        loss_fn = focal_loss
    model_dir.mkdir(parents=True, exist_ok=True)
    model.cuda()
    min_validation = torch.inf
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=3, normalize="true").cuda()
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=3)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x, y in tqdm(train_data_loader, desc=f"Training epoch {epoch}", total=len(train_data_loader)):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Train loss", epoch_loss / len(train_data_loader), epoch)
        writer.flush()

        print("Train loss " + str(epoch_loss / len(train_data_loader)))

        if epoch % 2 == 0:
            model.eval()
            confusion_matrix.reset()
            f1_score.reset()
            with torch.no_grad():
                val_loss = 0
                for x, y in tqdm(val_data_loader, desc=f"Validating epoch {epoch}", total=len(val_data_loader)):
                    x, y = x.cuda(), y.cuda()
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.item()
                    confusion_matrix(y_pred, torch.argmax(y, dim=1, keepdim=False))
                    f1_score(y_pred, torch.argmax(y, dim=1, keepdim=False))
                writer.add_scalar("Validation loss", val_loss / len(val_data_loader), epoch)
                writer.add_scalar("F1 score", f1_score.compute(), epoch)
                plot_confusion_matrix(confusion_matrix, writer, epoch)
            val_loss /= len(val_data_loader)
            if val_loss < min_validation:
                min_validation = val_loss
                torch.save(model.state_dict(), model_dir / f"best.h5")
                torch.save(model.state_dict(), model_dir / f"epoch{epoch}.h5")

            print("Validation loss " + str(val_loss))
            writer.flush()


def plot_confusion_matrix(confusion_matrix: torchmetrics.classification.MulticlassConfusionMatrix, writer: SummaryWriter, epoch: int) -> None:
    confusion_matrix_visualization = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix.compute().cpu().numpy(), display_labels=["background", "arch", "center"]
    )
    confusion_matrix_visualization.plot(
        xticks_rotation="vertical",
        im_kw={"vmin": 0, "vmax": 1},
        ax=plt.gca(),
        values_format=".2f",
    )
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    numpy_image = np.array(Image.open(buffer)) / 255
    writer.add_image("Confusion matrix", numpy_image[:, :, :3], epoch, dataformats="HWC")  # type: ignore[no-untyped-call]
    plt.close()
    buffer.close()


def focal_loss(preds, gt):  # type: ignore
    pos_inds = gt.gt(0.5)
    neg_inds = gt.le(0.5)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    preds_soft = F.softmax(preds, dim=1)
    pos_pred = preds_soft[pos_inds]
    neg_pred = preds_soft[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    loss = 0
    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
