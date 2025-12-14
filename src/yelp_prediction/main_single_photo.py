import torch
import train_single_photo

if __name__ == "__main__":
    # Load pre-computed features (using same features as MIL approach)
    features = torch.load("./data/features/features-clip.pt")

    # Train single-photo model
    train_single_photo.run(
        features_dict=features,
        input_dim=512,  # CLIP features
        epochs=20,
        batch_size=32,
    )
