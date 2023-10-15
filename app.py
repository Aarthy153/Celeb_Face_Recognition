import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load the trained model using pickle
with open('model\saved_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to process the uploaded image
def process_image(uploaded_image):
    # Convert the image to grayscale (if needed)
    uploaded_image = uploaded_image.convert('L')

    # Resize the image to match the input size expected by the model
    input_size = (64, 64)  # Adjust this to match the input size expected by your model
    uploaded_image = uploaded_image.resize(input_size)

    # Flatten the image to match the input shape expected by the model
    uploaded_image_array = np.array(uploaded_image).flatten().reshape(1, -1)

    # Predict using the trained model
    predicted_class = model.predict(uploaded_image_array)

    class_labels = {0: 'Serena Williams', 1: 'Lionel Messi', 2: 'Maria Sharapova'}
    predicted_celebrity = class_labels.get(predicted_class[0], 'Unknown')

    return predicted_celebrity

st.title('Sports Celebrity Recognition')

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Process the image and predict the celebrity
    predicted_celebrity = process_image(image)

    # Display the predicted celebrity
    st.write(f'Predicted Celebrity: {predicted_celebrity}')
