import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

st.title("🫁 Pneumonia Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    _, height, width, channels = model.input_shape

    image = image.resize((width, height))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    if confidence > 0.5:
        st.error(f"🚨 Pneumonia ({confidence * 100:.2f}%)")
    else:
        st.success(f"✅ Normal ({(1 - confidence) * 100:.2f}%)")