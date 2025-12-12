import torch
import torch.nn as nn


class MILModel(nn.Module):
    def __init__(
        self,
        *,
        median_stars=4.0,
        input_dim=1280,
    ):
        super().__init__()

        # Simple Regression Head
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        # careful: this is technically incorrect: we're using the median of the
        # whole dataset, not just our split
        initial_bias = torch.logit(torch.tensor((median_stars - 1) / 4))
        nn.init.constant_(self.head[-1].bias, initial_bias)

    def forward(self, x):
        # 1. MEAN POOLING (The "Bag" Aggregation)
        # Average across the K photos dimension
        bag_feature = torch.mean(x, dim=1)

        # 2. Regression
        raw_output = self.head(bag_feature)

        # 3. Exact match to RatingPredictor.py Sigmoid
        # Sigmoid -> [0, 1] * 4 -> [0, 4] + 1 -> [1, 5]
        return torch.sigmoid(raw_output) * 4 + 1
