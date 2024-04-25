from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Emotion Detection API", version="1.0")

# Cargar el modelo entrenado
model = load_model('model_optimal.h5')


@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Leer la imagen recibida en la petición como un flujo de bytes
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocesar la imagen para el modelo
        img = image.resize((48, 48))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = img_array / 255.0

        # Realizar predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
        predicted_emotion = emotion_dict[int(predicted_class)]

        return JSONResponse(content={"emotion": predicted_emotion}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)