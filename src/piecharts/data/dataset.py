import ast
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
import torch
import torchvision
from jaxtyping import Float, Int
from torch.utils.data import Dataset

from piecharts.utils.datatypes import Sector


class PiechartDataset(Dataset):
    dataframe: pd.DataFrame
    image_dir: Path

    def __init__(self, directory: Path, split: Literal["train", "val_and_test"]):
        self.dataframe = pd.read_csv(directory / f"{split}.csv")
        list_data_features = [
            "boxes",
            "start_angles",
            "end_angles",
            "angles",
            "percentages",
        ]
        for column in list_data_features:
            self.dataframe[column] = self.dataframe[column].apply(ast.literal_eval)
        self.dataframe["sectors"] = self.dataframe["boxes"].apply(lambda x: [Sector(*y) for y in x])
        self.image_dir = directory / "images" / "images"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> Tuple[Float[torch.Tensor, "3 X Y"], Int[torch.Tensor, "X Y"]]:
        image_path = self.image_dir / self.dataframe.iloc[index].filename
        image = torchvision.io.read_image(image_path)[:3] / 255

        mask = generate_mask(image.shape[1:], self.dataframe.sectors[index])
        return image, mask


def generate_mask(resolution: Tuple[int, int], sectors: List[Sector]):
    mask = torch.zeros(resolution).long()
    for sector in sectors:
        mask[sector.arc1_y, sector.arc1_x] = 1
        mask[sector.arc2_y, sector.arc2_x] = 1
        mask[sector.center_y, sector.center_x] = 2
    return mask


if __name__ == "__main__":
    dir = Path("data/raw")
    dataset = PiechartDataset(dir, "train")
    img, mask = dataset[10]
    torchvision.utils.save_image(img, "test.png")
    torchvision.utils.save_image(mask / 2, "test_mask.png")
