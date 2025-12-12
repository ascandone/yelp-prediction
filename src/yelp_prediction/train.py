import numpy as np
import torch
from dataframes import df_final, PHOTOS_DIR
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


BATCH_SIZE = 32  # T4 handles 32 easily with frozen backbone
LR = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YelpBagDataset(Dataset):
    DEFAULT_MAX_PHOTOS = 3  # Number of photos to sample per business

    # Standard ImageNet stats
    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        dataframe,
        photo_dir: Path,
        transform=YelpBagDataset.DEFAULT_TRANSFORM,
        max_photos=YelpBagDataset.DEFAULT_MAX_PHOTOS,
    ):
        self.photo_dir = photo_dir
        self.data = dataframe.to_dicts()
        self.transform = transform
        self.max_photos = max_photos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        photo_ids = row["photo_ids"]
        label = torch.tensor(row["stars"], dtype=torch.float32)

        # 1. SAMPLING: Pick K photos (Randomly sample if > K, repeat if < K)
        if len(photo_ids) >= self.max_photos:
            selected = np.random.choice(photo_ids, self.max_photos, replace=False)
        else:
            selected = np.random.choice(photo_ids, self.max_photos, replace=True)

        # Stack into [3, 3, 224, 224] tensor
        return torch.stack([self._load_photo(pid) for pid in selected]), label

    def _load_photo(self, pid):
        try:
            with Image.open(self.photo_dir / f"{pid}.jpg") as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img
        except Exception:
            # TODO use a neutral tensor
            # TODO emit warning
            # Fallback with a black placeholder image if image is missing/corrupt
            return torch.zeros(3, 224, 224)


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


def main():
    df = df_final

    median_stars = df["stars"].median()
    baseline_mae = (df["stars"] - median_stars).abs().mean()

    print(f"Baseline to beat: {baseline_mae:.2f} (median: {median_stars})")

    model = RatingPredictor(
        median_stars=median_stars,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Split 80/20

    mask = np.random.rand(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    photos_dir = PHOTOS_DIR / "photos"

    train_loader = DataLoader(
        YelpBagDataset(
            train_df,
            photo_dir=photos_dir,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        YelpBagDataset(
            val_df,
            photo_dir=photos_dir,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    print("Starting Training...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0

        # Progress bar for sanity check
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

        # --- QUICK EVAL ---
        model.eval()
        errors = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                errors.extend(torch.abs(preds - targets).cpu().numpy())

        print(f"--> Validation MAE: {np.mean(errors):.4f} stars\n")

    print("done!")
