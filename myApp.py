
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved Keras model
model = tf.keras.models.load_model('mnist_model.h5')

# Define the Streamlit app layout
st.title('MNIST Digit Classifier')

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 px)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_np = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize the image

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = model.predict(image_np)
    predicted_label = np.argmax(prediction)

    st.write(f'Predicted Label: {predicted_label}')

    # Visualize the prediction probabilities
    st.bar_chart(prediction[0])
