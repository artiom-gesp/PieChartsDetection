import argparse
from pathlib import Path

from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from piecharts.data.dataset import PiechartDataset
from piecharts.nn.models.config import TrainConfig
from piecharts.nn.models.smp_models import PSMModel
from piecharts.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parsed_args = parser.parse_args()

    with open(parsed_args.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = TrainConfig.model_validate(yaml)

    name = f"model_arch-{config.model.architecture}_encoder_{config.model.encoder_name}_freeze-{config.freeze_encoder}_center-{config.model.use_center_pool}_loss-{config.loss}"

    model = PSMModel(freeze_encoder=config.freeze_encoder, config=config.model)
    train_dataset = PiechartDataset(Path("data") / "raw", "train", "train", (256, 256))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = PiechartDataset(Path("data") / "raw", "train", "val", (256, 256))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model_dir = Path("models") / name
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_dir)  # type: ignore

    json_config = config.model_dump_json(indent=4)
    with (model_dir / "config.json").open("w", encoding="utf8") as f:
        f.write(json_config)

    train(model, train_dataloader, val_dataloader, writer, config.epochs, model_dir=model_dir, lr=config.lr, loss=config.loss)


if __name__ == "__main__":
    main()
