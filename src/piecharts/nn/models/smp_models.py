from functools import partial
from typing import Optional

import segmentation_models_pytorch as smp
import torch
from jaxtyping import Float
from torch import nn

from piecharts.nn.models.config import ModelConfig


class PSMModel(nn.Module):
    base: nn.Module
    preprocessing_function: partial
    center_pool: Optional[nn.Module] = None

    def __init__(self, config: ModelConfig, freeze_encoder: bool, **kwargs: str) -> None:
        super().__init__()

        self.base = smp.create_model(config.architecture, classes=3, **kwargs)
        self.preprocessing_function = smp.encoders.get_preprocessing_fn(config.encoder_name)

        if freeze_encoder:
            for param in self.base.encoder.parameters():
                param.requires_grad = False

        dim = kwargs["decoder_channels"][-1] if "decoder_channels" in kwargs else 16
        if config.use_center_pool:
            from piecharts.nn.layers.center_pool import CenterPoolingLayer

            self.center_pool = CenterPoolingLayer(dim)

    def forward(self, x: Float[torch.Tensor, "B 3 X Y"]) -> Float[torch.Tensor, "B 3 X Y"]:
        features = self.base.encoder(x)
        x = self.base.decoder(*features)
        if self.center_pool:
            x = self.center_pool(x)
        return self.base.segmentation_head(x)
