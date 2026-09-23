from torch import nn
from torchvision import models


def build_mobilenet_v3_small(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
