import io
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import uvicorn
from model import SinglePhotoModel
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# TODO get from env
MODEL_PATH = Path("data/best_model.pth")
DEVICE = torch.device("cpu")


def _get_model():
    model = SinglePhotoModel(input_dim=512)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# Load model once globally
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=False,
)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

rating_model = _get_model()


def _predict(post_data: bytes) -> float:
    image = Image.open(io.BytesIO(post_data)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        prediction_tensor = rating_model(outputs)
        return prediction_tensor.item()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def upload_photo(file: UploadFile = File(...)):
    file_bytes = await file.read()
    score = _predict(file_bytes)
    return {
        "score": score,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
