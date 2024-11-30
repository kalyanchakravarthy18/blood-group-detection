import streamlit as st
import gdown
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def download_model(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file. Unable to read using OpenCV.")
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = img.reshape(1, 128, 128, 1)
        return img
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return None

def predict_blood_group(model, label_mapping, image_array):
    prediction = np.argmax(model.predict(image_array), axis=1)[0]
    blood_group = [key for key, value in label_mapping.items() if value == prediction][0]
    return blood_group

def main():
    st.title("Blood Group Detection App")
    st.write("Upload a fingerprint image to predict the blood group.")

    file_id = "1-2hw5VPj1p2yEdykbWshtbW6jC_7zEXY"
    model_path = "bloodgroup.keras"

    label_mapping = {
        "A+": 0,
        "A-": 1,
        "AB+": 2,
        "AB-": 3,
        "B+": 4,
        "B-": 5,
        "O+": 6,
        "O-": 7,
    }

    if not os.path.exists(model_path):
        st.write("Downloading the model...")
        try:
            download_model(file_id, model_path)
            st.success(f"Model downloaded to: {model_path}")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return
    else:
        st.write(f"Model found at: {model_path}")

    try:
        st.write("Loading the model...")
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    uploaded_file = st.file_uploader("Choose a fingerprint image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)

        preprocessed_image = preprocess_image(temp_image_path)

        if preprocessed_image is not None:
            st.write("Running prediction...")
            with st.spinner("Predicting..."):
                try:
                    blood_group = predict_blood_group(model, label_mapping, preprocessed_image)
                    st.success(f"Predicted Blood Group: {blood_group}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
