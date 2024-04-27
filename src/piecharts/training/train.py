from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model: nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, writer: SummaryWriter, epochs: int, model_dir: Path, lr: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model_dir.mkdir(parents=True, exist_ok=True)
    model.cuda()
    min_validation = torch.inf

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
            with torch.no_grad():
                val_loss = 0
                for x, y in tqdm(val_data_loader, desc=f"Validating epoch {epoch}", total=len(val_data_loader)):
                    x, y = x.cuda(), y.cuda()
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.item()
                writer.add_scalar("Validation loss", val_loss / len(val_data_loader), epoch)
            val_loss /= len(val_data_loader)
            if val_loss < min_validation:
                min_validation = val_loss
                torch.save(model.state_dict(), model_dir / f"best.h5")
                torch.save(model.state_dict(), model_dir / f"epoch{epoch}.h5")

            print("Validation loss " + str(val_loss))
            writer.flush()

