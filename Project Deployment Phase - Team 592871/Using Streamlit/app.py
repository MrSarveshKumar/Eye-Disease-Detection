# app.py
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the trained model
model = load_model('EyeModel.h5')

# Define the class labels
class_labels = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Glaucoma', 3: 'Normal'}

# Streamlit app
st.title("Eye Disease Detection")
st.write("Upload an image to detect the disease.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file is uploaded before attempting to use it
if uploaded_file is not None:
    plt.axis('off')
    plt.imshow(np.array(Image.open(uploaded_file)))
    st.pyplot()

    if st.button("Predict"):
        img = image.load_img(uploaded_file, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x))

        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"The model predicts: {class_labels[pred]}")
else:
    st.write("Please upload an image before predicting.")

st.markdown("****")

st.write("NOTE: This is only for Educational Purpose")
st.write("<span style='font-size: 15px;'>Founder: *Sarvesh Kumar*</span>", unsafe_allow_html=True)
