import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='features.35'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:36])
        self.layer_name = layer_name
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # Ensure mean and std are on the same device as x
        device = x.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        x = (x - self.mean) / self.std
        features = self.features(x)
        return features

def perceptual_loss_fn(y_true, y_pred):
    feature_extractor = VGGFeatureExtractor().to(y_true.device)
    y_true_features = feature_extractor(y_true)
    y_pred_features = feature_extractor(y_pred)
    return nn.MSELoss()(y_true_features, y_pred_features)
