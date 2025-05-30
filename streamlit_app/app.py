# Streamlit app main file
# streamlit_app/app.py
import streamlit as st
import os
from PIL import Image
import numpy as np
from utils.predict_cnn import predict_disease
from utils.predict_yolo import detect_weeds

st.set_page_config(page_title="Plant Disease & Weed Detection", layout="wide")
st.title("🌿 Plant Disease Detection and Weed Localization")

# Upload multiple images
uploaded_files = st.file_uploader("Upload Leaf or Field Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Batch Processing Results")
    for uploaded_file in uploaded_files:
        st.markdown("---")
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption=uploaded_file.name, use_column_width=True)

        # Disease prediction
        disease_result = predict_disease(img_array)
        st.success(f"Disease Detection (CNN): {disease_result}")

        # Weed detection using YOLO
        output_img = detect_weeds(img_array)
        st.image(output_img, caption="Weed Localization (YOLOv5)", use_column_width=True)

st.markdown("---")
st.info("This app uses trained CNN for plant disease classification and YOLOv5 for weed localization. GAN was used for synthetic image generation during training.")

