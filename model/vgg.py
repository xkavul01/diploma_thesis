from typing import Optional, Any

import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, VGG
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.vgg import make_layers, cfgs


class VGGOCR(VGG):
    def __init__(self,
                 features: nn.Module,
                 num_classes: int = 1000,
                 init_weights: bool = True,
                 dropout: float = 0.5
                 ) -> None:
        super().__init__(features, num_classes, init_weights, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(23): # 23
            x = self.features[i](x)
        return x


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGGOCR(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def vgg16_bn(*, weights: Optional[VGG16_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, **kwargs)
