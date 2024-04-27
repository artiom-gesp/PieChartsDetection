import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model: nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, writer: SummaryWriter, epochs: int):
    optimizer = torch.optim.Adam(model.parameters)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.cuda()

    for epoch in epochs:
        epoch_loss = 0
        model.train()
        for x, y in tqdm(train_data_loader, desc=f"Training epoch {epoch}", total=len(train_data_loader)):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            loss_fn.backward()
            optimizer.step()
        writer.add_scalar("Train loss", epoch_loss, epoch)

        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x, y in tqdm(val_data_loader, desc=f"Validating epoch {epoch}", total=len(val_data_loader)):
                    x, y = x.cuda(), y.cuda()
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.item()
                writer.add_scalar("Validation loss", val_loss, epoch)
