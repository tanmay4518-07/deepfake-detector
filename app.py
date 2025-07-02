import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("model/deepfake_model.h5")

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    label = "Fake" if pred > 0.5 else "Real"
    confidence = f"{pred*100:.2f}%" if label == "Fake" else f"{(1 - pred)*100:.2f}%"
    return f"{label} ({confidence})"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Label(label="Prediction"),
    title="DeepFake Detection",
    description="Upload an image to check if it's Real or Fake (AI-generated).",
    allow_flagging="never"
)

interface.launch()
