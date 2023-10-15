import streamlit as st
import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load('saved_model.pkl')

# Define a function to process the uploaded image
def process_image(uploaded_image):
    # Convert the image to grayscale (if needed)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the input size expected by the model
    input_size = (64, 64)  # Adjust this to match the input size expected by your model
    uploaded_image = cv2.resize(uploaded_image, input_size)

    # Flatten the image to match the input shape expected by the model
    uploaded_image_flattened = uploaded_image.flatten().reshape(1, -1)

    # Predict using the trained model
    predicted_class = model.predict(uploaded_image_flattened)

    class_labels = {0: 'Serena Williams', 1: 'Lionel Messi', 2: 'Maria Sharapova'}
    predicted_celebrity = class_labels.get(predicted_class[0], 'Unknown')

    return predicted_celebrity

st.title('Sports Celebrity Recognition')

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Process the image and predict the celebrity
    predicted_celebrity = process_image(image)

    # Display the predicted celebrity
    st.write(f'Predicted Celebrity: {predicted_celebrity}')
