import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from utils.preprocessing import preprocess_audio

st.set_page_config(page_title="CoughNet")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_cnn_model.keras"
    )

model = load_model()

st.title("CoughNet")
st.write(
    "Deep Learning-Based Respiratory Condition Screening"
)

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    mel_img = preprocess_audio(uploaded_file)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(
        mel_img.squeeze(),
        aspect="auto",
        origin="lower"
    )
    ax.set_title("Mel Spectrogram")

    st.pyplot(fig)

    pred = model.predict(
        np.expand_dims(mel_img, axis=0),
        verbose=0
    )[0][0]

    probability = float(pred)

    st.subheader("Prediction")

    if probability >= 0.30:
        st.error(
            f"Symptomatic / COVID-like cough "
        )
    else:
        st.success(
            f"Healthy "
        )

    # st.write(
    #     f"Model Probability: {probability:.4f}"
    # )