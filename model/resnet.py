from typing import Optional, Any, Type, Union, List

from torch import Tensor
from torchvision.models import ResNet, ResNet50_Weights, ResNet18_Weights, ResNet34_Weights
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetOCR(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 **kwargs: Any
                 ):
        super().__init__(block, layers, **kwargs)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetOCR(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
