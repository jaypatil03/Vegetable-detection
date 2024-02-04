import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Image Classification App",
    page_icon=":camera:",
    layout="centered",
)

st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make a request to the FastAPI server for prediction
    files = {"file": uploaded_file}
    response = requests.post("http://your-fastapi-server-url/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: Class - {result['class']}, Confidence - {result['confidence']:.2f}")
    else:
        st.error("Error occurred during classification. Please try again.")
