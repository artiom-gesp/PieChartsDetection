from functools import partial
from typing import Optional

import segmentation_models_pytorch as smp
import torch
from jaxtyping import Float
from torch import nn


class PSMModel(nn.Module):
    base: nn.Module
    preprocessing_function: partial
    center_pool: Optional[nn.Module] = None

    def __init__(self, use_center_pool: bool, arch: str = "unet", encoder_name: str = "resnet34", **kwargs) -> None:
        super().__init__()

        self.base = smp.create_model(arch, classes=3, **kwargs)
        self.preprocessing_function = smp.encoders.get_preprocessing_fn(encoder_name)

        for param in self.base.encoder.parameters():
            param.requires_grad = False

        dim = kwargs["decoder_channels"][-1] if "decoder_channels" in kwargs else 16
        if use_center_pool:
            from piecharts.nn.layers.center_pool import CenterPoolingLayer

            self.center_pool = CenterPoolingLayer(dim)

    def forward(self, x: Float[torch.Tensor, "B 3 X Y"]) -> Float[torch.Tensor, "B 3 X Y"]:
        x = self.preprocessing_function(x)
        x = self.base.encoder(x)
        x = self.base.decoder(x)
        if self.center_pool:
            x = self.center_pool(x)
        return self.base.segmentation_head(x)
