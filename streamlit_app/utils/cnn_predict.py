# CNN prediction utilities
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained CNN model
MODEL_PATH = os.path.join("app", "models", "cnn_disease_model.h5")
cnn_model = load_model(MODEL_PATH)

# Define class names in the same order as used in training
CLASS_NAMES = ['healthy', 'tomato_disease', 'mulberry_disease', 'carrot_disease']

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = cnn_model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence
