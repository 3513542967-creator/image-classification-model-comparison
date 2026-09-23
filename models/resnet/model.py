from torch import nn
from torchvision import models


def build_resnet(name, num_classes):
    constructors = {"resnet18": models.resnet18, "resnet34": models.resnet34}
    if name not in constructors:
        raise ValueError(f"Unknown ResNet: {name}")
    model = constructors[name](weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
