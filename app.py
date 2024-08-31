import streamlit as st
import tensorflow as tf
from tensorflow.keras import models , layers
from PIL import Image
import numpy as np




#Function--->
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [256, 256])  # Resize to match training dimensions  # Normalize the image to [0, 1] range
    return image

# Function predictions
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Load Models --->
# Define the class names for the potato diseases
class_names = ["Early Blight", "Late Blight", "Healthy"]
model_paths = {
    "Model 1 (CNN)": "https://github.com/sadia4444a/Potato-Disease-Classification/blob/main/models/Potato_v1.keras",
    # "Model 2": "path/to/other_model_2.keras",
    # "Model 3": "path/to/other_model_3.keras",
    # "Model 4": "path/to/other_model_4.keras",
    # "Model 5": "path/to/other_model_5.keras",
    # "Model 6": "path/to/other_model_6.keras",
}




#streamlit app design->>>>>>

st.sidebar.title("Select a Model")

selected_model_name = st.sidebar.selectbox("Choose a model for prediction:", list(model_paths.keys()))
selected_model_path = model_paths[selected_model_name]
# Load the selected model
model = tf.keras.models.load_model(selected_model_path)



# Title of the app
st.title(":red[ 🥔🍃 Potato leaf  Disease Classification]")
st.write("""
This app classifies potato leaves as either **Healthy**, **Early Blight**, or **Late Blight**.
Select a model from the sidebar to start.
""")

st.image('potato_leaf_disease.png', caption='potato leaf image ')

# File uploader for image
uploaded_file = st.file_uploader("Upload a potato leaf image...", type=["jpg", "jpeg", "png"])


# Predict button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
     # Convert the image to numpy array
    image=preprocess_image(image)
    if st.button("Predict", type='primary'):
        predicted_class, confidence = predict(model, image.numpy())
        # Display the results
        st.write(f"**Predicted Class:** :red[{predicted_class}]")
        st.write(f"**Confidence:** :red[{confidence:.2f}%]")

else:
    st.write("Please upload an image to get a prediction.")