import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# load model
model = load_model('classification\model\model.h5')

# Untuk memprediksi gambar
def predict_image(image, model, classes):
    img = image.resize((125, 125))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_idx]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence, predictions[0]

# urutan kelas
classes = [
    "cables",       
    "case",         
    "cpu",           
    "gpu",           
    "hdd",           
    "headset",       
    "keyboard",      
    "microphone",    
    "monitor",       
    "motherboard",   
    "mouse",        
    "ram",          
    "speakers",     
    "webcam"        
]

st.title("AI Web")
st.subheader("Klasifikasi Komponen Komputer dengan Model AI")

# Unggah Image
uploaded_image = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    with st.spinner("Memproses gambar..."):
        # Hasil prediksi
        predicted_class, confidence, probabilities = predict_image(image, model, classes)

    # Display results
    st.success(f"Prediksi: {predicted_class}")
    st.info(f"Confidence Score: {confidence:.2f}%")

    # Visualization
    st.subheader("Visualisasi Hasil Prediksi")
    fig, ax = plt.subplots()
    ax.bar(classes, probabilities)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    st.pyplot(fig)
