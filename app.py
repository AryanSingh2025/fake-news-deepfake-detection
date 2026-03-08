import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Fake News Model
fake_model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load Deepfake Model
deepfake_model = tf.keras.models.load_model("models/deepfake_model.h5")

st.title("Fake News + Deepfake Detection System")

option = st.selectbox(
    "Choose Detection Type",
    ("Fake News Detection", "Deepfake Detection")
)

# Fake News Section
if option == "Fake News Detection":

    news = st.text_area("Enter News Article")

    if st.button("Check News"):

        if news:

            news_vector = vectorizer.transform([news])
            prediction = fake_model.predict(news_vector)

            if prediction[0] == 0:
                st.error("⚠️ Fake News Detected")

            else:
                st.success("✅ Real News")

# Deepfake Section
if option == "Deepfake Detection":

    image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if image:

        img = Image.open(image)
        st.image(img)

        img = img.resize((128,128))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = deepfake_model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.error("⚠️ Deepfake Image Detected")
        else:
            st.success("✅ Real Image")