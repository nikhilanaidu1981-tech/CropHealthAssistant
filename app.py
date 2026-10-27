
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="Crop Health Detection Assistant",
    page_icon="🌱"
)

st.title("🌱 Crop Health Detection Assistant")
st.write("Upload a leaf image to check its predicted condition.")

REPO = "Nefflymicn/PlantVillage-plant-disease-detection"

@st.cache_resource
def load_ai_model():
    model_path = hf_hub_download(
        repo_id=REPO,
        filename="plant_disease_efficientnet.keras"
    )
    labels_path = hf_hub_download(
        repo_id=REPO,
        filename="class_names.txt"
    )

    model = load_model(model_path)

    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, name = line.strip().split(":", 1)
                labels[int(idx.strip())] = name.strip()

    return model, labels

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    if st.button("Detect Leaf Disease"):
        try:
            with st.spinner("Loading AI model and analyzing image..."):
                model, labels = load_ai_model()

                image = image.resize((224, 224))
                array = np.array(image, dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                array = preprocess_input(array)

                probabilities = model.predict(array, verbose=0)[0]
                index = int(np.argmax(probabilities))
                result = labels.get(index, f"Class {index}")
                confidence = float(probabilities[index]) * 100

            st.subheader("Prediction")
            st.write(result.replace("___", " — ").replace("_", " "))
            st.metric("Model confidence", f"{confidence:.1f}%")

            if "healthy" in result.lower():
                st.success("The model predicts this leaf may be healthy.")
            else:
                st.warning(
                    "The model predicts a possible disease. "
                    "Please consult an agricultural expert to confirm."
                )

            st.caption(
                "Experimental AI result only. This model may be inaccurate "
                "on real-world photos and must not be treated as a diagnosis."
            )

        except Exception as error:
            st.error("The AI model could not complete the prediction.")
            st.code(str(error))
