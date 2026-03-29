from pathlib import Path
from io import BytesIO

import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="Garbage Classification API")

MODEL_DIR = Path("models/cnn_improved")
LABELS_PATH = MODEL_DIR / "labels.txt"
WEIGHTS_PATH = MODEL_DIR / "best_model.pt"

if not LABELS_PATH.exists():
    raise RuntimeError("labels.txt introuvable dans models/cnn_improved")

if not WEIGHTS_PATH.exists():
    raise RuntimeError("best_model.pt introuvable dans models/cnn_improved")

labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/")
def root():
    return {"message": "API garbage classification OK"}


@app.get("/health")
def health():
    return {"status": "ok", "num_classes": len(labels)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    content = await file.read()

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide.")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {
        labels[i]: float(probs[i])
        for i in range(len(labels))
    }

    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": all_probs,
    }