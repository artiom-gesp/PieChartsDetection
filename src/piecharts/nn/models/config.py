from typing import Tuple

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    architecture: str = Field("unet")
    encoder_name: str = Field("resnet34")
    use_center_pool: bool = Field(True)


class TrainConfig(BaseModel):
    batch_size: int = Field(32)
    lr: float = Field(0.001)
    resolution: Tuple[int, int] = Field((256, 256))
    freeze_encoder: bool = Field(True)
    model: ModelConfig = Field(default_factory=ModelConfig)
    epochs: int = Field(100)