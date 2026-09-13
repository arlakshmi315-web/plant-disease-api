from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI(title="Plant Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and supporting files once at startup
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "plant_disease_model.h5"))

with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

with open(os.path.join(BASE_DIR, "disease_info.json")) as f:
    disease_info = json.load(f)


@app.get("/")
async def root():
    return {"message": "Plant Disease Prediction API is running!"}


@app.post("/predict")
async def predict(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    class_name = labels[idx]
    confidence = float(preds[0][idx])

    return {
        "disease": class_name,
        "confidence": round(confidence * 100, 2),
        "info": disease_info.get(class_name, {})
    }
