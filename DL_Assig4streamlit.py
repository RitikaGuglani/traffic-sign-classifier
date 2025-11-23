import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("traffic_classifier.h5")


classes = [str(i) for i in range(43)]

st.title("Traffic Sign Classification Demo")
st.write("Upload an image and the model will predict the traffic sign class.")

uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (64, 64))
    st.image(img, caption="Uploaded Image", channels="BGR")

    
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    
    prediction = model.predict(img_input)
    pred_class = np.argmax(prediction)

    st.success(f"Predicted Class: {pred_class}")
