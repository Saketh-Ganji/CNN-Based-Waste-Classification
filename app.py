# waste_classifier_app/app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Class labels
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model
@st.cache_resource

def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

st.title("Garbage Classifier")
st.write("Upload an image of garbage and the model will classify it into one of six categories:")
st.write("**cardboard, glass, metal, paper, plastic, trash**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    predictions = model.predict(x)
    class_index = np.argmax(predictions[0])
    st.write(f"Predicted Class: **{CLASS_NAMES[class_index]}**")
