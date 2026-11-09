import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_medical_model(num_classes=1, freeze_base=True, pretrained=True):
    """ResNet50 + small head; pretrained=False skips the ImageNet download."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model
