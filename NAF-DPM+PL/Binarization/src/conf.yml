import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='features.16'):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:int(layer_name.split('.')[1]) + 1])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

