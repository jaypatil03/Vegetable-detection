import streamlit as st
import requests

# Streamlit app title
st.title("Image Classification with FastAPI and Streamlit")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        # Send image to FastAPI server for prediction
        files = {"file": uploaded_file.getvalue()}
        
        # Update the URL to the deployed FastAPI server on Streamlit Cloud
        response = requests.post("https://your-streamlit-app-url.herokuapp.com/predict", files=files)

        # Display prediction result
        if response.ok:
            prediction_data = response.json()
            st.success(f"Class: {prediction_data['class']}, Confidence: {prediction_data['confidence']:.2f}")
        else:
            st.error("Error occurred during prediction. Please try again.")
