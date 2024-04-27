import ast
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd
import torch
import torchvision
from jaxtyping import Float, Int
from torch.utils.data import Dataset

from piecharts.utils.datatypes import Sector


class PiechartDataset(Dataset):
    dataframe: pd.DataFrame
    image_dir: Path
    resolution: Optional[Tuple[int, int]]
    split: str
    seeds: Optional[Int[torch.Tensor, "N"]] = None

    def __init__(
        self, directory: Path, split: Literal["train", "val_and_test"], train_split: Literal["train", "val"], resolution: Optional[Tuple[int, int]] = None
    ):
        self.dataframe = pd.read_csv(directory / f"{split}.csv")
        if split == "train":
            if train_split == "train":
                self.dataframe = self.dataframe.iloc[:8000].reset_index(drop=True)
            else:
                self.dataframe = self.dataframe.iloc[8000:].reset_index(drop=True)

        if split == "val_and_test" or train_split == "val":
            self.seeds = torch.randint(0, 1000, (len(self.dataframe),))

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
        self.resolution = resolution

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[Float[torch.Tensor, "3 X Y"], Int[torch.Tensor, "3 X Y"]]:
        with torch.random.fork_rng(enabled=self.seeds is not None):
            if self.seeds is not None:
                torch.random.manual_seed(self.seeds[index])

            image_path = self.image_dir / self.dataframe.iloc[index].filename
            image = torchvision.io.read_image(image_path)[:3] / 255

            mask = generate_mask(image.shape[1:], self.dataframe.sectors[index])

            if self.resolution is not None:
                if mask.shape[1] <= self.resolution[0] or mask.shape[2] <= self.resolution[1]:
                    scale = max((self.resolution[0] + 1) / mask.shape[1], (self.resolution[1] + 1) / mask.shape[2])
                    image = torch.nn.functional.interpolate(image[None], scale_factor=scale, mode="bicubic")[0]
                    mask = torch.nn.functional.interpolate(mask[None], scale_factor=scale, mode="nearest-exact")[0]

                x_start = torch.randint(0, int(mask.shape[1] - self.resolution[0] + 1), (1,))
                y_start = torch.randint(0, int(mask.shape[2] - self.resolution[1] + 1), (1,))
                image = image[:, x_start : x_start + self.resolution[0], y_start : y_start + self.resolution[1]]
                mask = mask[:, x_start : x_start + self.resolution[0], y_start : y_start + self.resolution[1]]

            return image, mask


def generate_mask(resolution: Tuple[int, int], sectors: List[Sector]) -> Float[torch.Tensor, "3 X Y"]:
    mask = torch.zeros((3, *resolution))

    kernel_size = (5, 5)
    sigma = 1
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = (kernel_size[0] // 2, kernel_size[1] // 2)

    for sector in sectors:
        points = [(1, sector.arc1_x, sector.arc1_y), (1, sector.arc2_x, sector.arc2_y), (2, sector.center_x, sector.center_y)]
        for layer, x, y in points:
            x_min = max(0, x - pad_size[0])
            x_max = min(mask.size(2), x + pad_size[0] + 1)
            y_min = max(0, y - pad_size[1])
            y_max = min(mask.size(1), y + pad_size[1] + 1)
            kernel_x_min = pad_size[0] - (x - x_min)
            kernel_x_max = kernel_x_min + (x_max - x_min)
            kernel_y_min = pad_size[1] - (y - y_min)
            kernel_y_max = kernel_y_min + (y_max - y_min)
            mask[layer, y_min:y_max, x_min:x_max] += kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]

    mask = torch.clamp(mask, 0, 1)

    mask[0:1] += 1 - mask.sum(dim=0, keepdim=True)  # Ensure sum over channels is 1
    return mask


def gaussian_kernel(size: Tuple[int, int] = (5, 5), sigma: float = 2.0) -> Float[torch.Tensor, "N N"]:
    m, n = size
    y, x = torch.meshgrid(torch.arange(m), torch.arange(n), indexing="ij")
    y0, x0 = m // 2, n // 2
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


if __name__ == "__main__":
    dir = Path("data/raw")
    dataset = PiechartDataset(dir, "train", (256, 256))
    for i in range(2):
        img, mask = dataset[i]
        torchvision.utils.save_image(img, f"test{i}.png")
        torchvision.utils.save_image(mask, f"test_mask{i}.png")
