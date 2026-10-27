
import streamlit as st

st.set_page_config(
    page_title="Crop Health Detection Assistant",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Health Detection Assistant")
st.write(
    "Upload a crop leaf image to start your "
    "plant health assessment."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded crop leaf",
        use_container_width=True
    )

    st.info(
        "Image uploaded successfully! "
        "AI disease prediction will be added next."
    )

st.warning(
    "This prototype does not diagnose plant diseases yet. "
    "Consult an agricultural expert if you notice symptoms."
)
