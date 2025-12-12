import torch
from torchvision import models
import torch.nn as nn


class RatingPredictor(nn.Module):
    def __init__(self, median_stars):
        super().__init__()
        # Load ResNet50 with modern weights
        base_model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        # Strip the last layer (fc)
        self.backbone = base_model.features

        # FREEZE BACKBONE (Crucial for speed)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),  # Output 1 star rating
        )

        # careful: this is technically incorrect: we're using the median of the
        # whole dataset, not just our split
        initial_bias = torch.logit(torch.tensor((median_stars - 1) / 4))
        nn.init.constant_(self.head[-1].bias, initial_bias)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x shape: [Batch, K_Photos, Channels, H, W]
        b, k, c, h, w = x.shape

        # Flatten batch and K dimensions to pass through ResNet
        x = x.view(b * k, c, h, w)

        x = self.backbone(x)

        # Features: [B*K, 1280, 1, 1] -> [B*K, 1280]
        # Reshape back to [B, K, Features]
        features = self.pool(x).squeeze().view(b, k, -1)

        # MEAN POOLING: Average features of the 3 photos
        avg_features = torch.mean(features, dim=1)

        # we force the output to be 1<=x<=5
        raw_output = self.head(avg_features).squeeze()
        return torch.sigmoid(raw_output) * 4 + 1
