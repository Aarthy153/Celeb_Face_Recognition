
import streamlit as st
import cv2
import numpy as np
import joblib

model = joblib.load('saved_model.pkl')

def process_image(uploaded_image):
    st.write("Processing image...")

    # Resize the image to match the input size expected by the model
    input_size = (64, 64)  # Adjust this to match the input size expected by your model
    uploaded_image = cv2.resize(uploaded_image, input_size)

    # Display the processed image
    st.image(uploaded_image, caption='Processed Image.', use_column_width=True)

    # Flatten the image to match the input shape expected by the model
    uploaded_image_flattened = uploaded_image.flatten().reshape(1, -1)

    # Display the flattened image
    st.write("Flattened Image:", uploaded_image_flattened)

    # Predict using the trained model
    predicted_class = model.predict(uploaded_image_flattened)

    # Map the predicted class to a human-readable label (e.g., celebrity name)
    # Replace this with your own mapping
    class_labels = {'croppedserena_williams': 0,'croppedroger_federer': 1,'croppedlionel_messi': 2,
                    'croppedvirat_kohli': 3,  'croppedmaria_sharapova': 4}
    predicted_celebrity = class_labels.get(predicted_class[0], 'Unknown')

    return predicted_celebrity
