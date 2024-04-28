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





class BetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(3, 50, 3, padding=1)
        self.ln1 = torch.nn.BatchNorm2d(50)
        self.down1 = torch.nn.Conv2d(50, 100, 3, stride=2, padding=1)
        self.l2 = torch.nn.Conv2d(100, 150, 3, padding=1)
        self.ln2 = torch.nn.BatchNorm2d(150)
        self.down2 = torch.nn.Conv2d(150, 200, 3, stride=2, padding=1)
        self.l3 = torch.nn.Conv2d(200, 200, 3, padding=1)
        self.ln3 = torch.nn.BatchNorm2d(200)
        self.l4 = torch.nn.Conv2d(200, 200, 3, padding=1) 
        self.up1 = torch.nn.ConvTranspose2d(200, 100, 2, stride=2)
        self.l5 = torch.nn.Conv2d(250, 100, 3, padding=1)
        self.ln4 = torch.nn.BatchNorm2d(100)
        self.up2 = torch.nn.ConvTranspose2d(100, 50, 2, stride=2)
        self.l6 = torch.nn.Conv2d(100, 50, 3, padding=1)
        self.ln5 = torch.nn.BatchNorm2d(50)
        self.l7 = torch.nn.Conv2d(50, 3, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.ln1(self.l1(x)))
        y = torch.relu(self.down1(x))
        y = torch.relu(self.ln2(self.l2(y)))
        z = torch.relu(self.down2(y))
        z = torch.relu(self.ln3(self.l3(z)))
        z = torch.relu(self.l4(z))
        w = torch.relu(self.up1(z))
        w = torch.concatenate([w, y], 1)
        w = torch.relu(self.ln4(self.l5(w)))
        o = torch.relu(self.up2(w))
        o = torch.concatenate([o, x], 1)
        p = torch.relu(self.ln5(self.l6(o)))
        
        return self.l7(p)



