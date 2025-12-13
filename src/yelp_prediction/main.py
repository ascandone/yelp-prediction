import torch
import train_features

if __name__ == "__main__":
    features = torch.load("./data/features/features-clip.pt")
    train_features.run(
        features_dict=features,
        max_photos=5,
        input_dim=512,
    )
