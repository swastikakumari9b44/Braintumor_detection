from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
import cv2
from gradcam import get_gradcam
import base64
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.h5")

def preprocess(image):
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_array = preprocess(image)

    pred = model.predict(img_array)[0][0]
    label = "Tumor" if pred > 0.5 else "No Tumor"
    confidence = float(pred) if pred > 0.5 else float(1 - pred)

    # Grad-CAM
    heatmap = get_gradcam(model, img_array)

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    _, buffer = cv2.imencode('.jpg', heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "prediction": label,
        "confidence": confidence,
        "heatmap": heatmap_base64
    }