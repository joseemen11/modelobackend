#!/usr/bin/env python3

from pathlib import Path
import io

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow import keras
from PIL import Image

# -----------------------------------------------
# 1) Carga del modelo (misma carpeta del script)
# -----------------------------------------------
MODEL_PATH = Path(__file__).parent / "modelo_resnet_entrenado.keras"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
model = keras.models.load_model(str(MODEL_PATH))

# Etiquetas de FER2013
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -----------------------------------------------
# 2) Función de preprocesado de bytes a tensor
# -----------------------------------------------
def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

# -----------------------------------------------
# 3) Definición de la app FastAPI
# -----------------------------------------------
app = FastAPI(
    title="Detector de Emociones",
    description="API que recibe una imagen y devuelve la emoción predominante",
    version="1.0",
)

# Configuración CORS para localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ⚠️ permite cualquier origen (útil en dev)
    allow_credentials=True,
    allow_methods=["*"],       # todos los métodos HTTP
    allow_headers=["*"],       # todos los headers
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validación simple
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Solo se permiten archivos de imagen")
    # Lectura y preprocesado
    img_bytes = await file.read()
    x = preprocess_bytes(img_bytes)
    # Inferencia
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    # Construcción de respuesta
    result = {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(preds[idx]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)},
    }
    return JSONResponse(result)

# -----------------------------------------------
# 4) Lanzamiento del servidor
# -----------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "modelo:app",       # "<nombre del archivo sin .py>:<nombre de la variable>"
        host="0.0.0.0",
        port=8000,
        reload=True         # ahora sí funciona correctamente
    )