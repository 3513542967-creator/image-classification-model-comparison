"""Model factory and lightweight parameter/FLOPs reporting."""
import torch
from torch import nn
from .resnet import build_resnet
from .mobilenetv3 import build_mobilenet_v3_small
from .vit import VisionTransformer


def build_model(name: str, num_classes: int) -> nn.Module:
    if name.startswith("resnet"):
        return build_resnet(name, num_classes)
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(num_classes)
    if name == "vit_tiny":
        return VisionTransformer(num_classes)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_macs(model: nn.Module, image_size: int = 224) -> int:
    """Estimate MACs from convolution/linear operations and attention blocks."""
    total = 0
    hooks = []
    def add_macs(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            batch, out_channels, height, width = output.shape
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += batch * out_channels * height * width * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
        elif isinstance(module, nn.MultiheadAttention):
            tokens = inputs[0].shape[1]
            batch, embed = inputs[0].shape[0], module.embed_dim
            total += batch * (4 * tokens * embed * embed + 2 * tokens * tokens * embed)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(add_macs))
    try:
        model.eval()( torch.zeros(1, 3, image_size, image_size))
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)
