import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Streamlit page config ---
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="centered"
)

# --- Load model ---
model = load_model('teeth_model.h5')

# --- Define class names and icons ---
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
class_icons = {
    'CaS': '🦷',    # Caries Superficial
    'CoS': '⚠️',    # Caries Severe
    'Gum': '🌿',     # Gum disease
    'MC': '💊',      # Mucosal Conditions
    'OC': '🔥',      # Oral Cancer
    'OLP': '🧩',    # Oral Lichen Planus
    'OT': '❓'       # Other conditions
}

# --- Streamlit UI ---
st.title("🦷 Teeth Disease Classification")
st.write("Upload an image of a tooth and see AI predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]
    predicted_class = class_names[class_idx]
    predicted_icon = class_icons.get(predicted_class, '❓')

    # Create gradient color based on confidence
    green = int(confidence * 255)
    red = 255 - green
    gradient_color = f"rgba({red}, {green}, 100, 0.3)"

    # Display result with style
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.0), {gradient_color});
        padding:25px;
        border-radius:20px;
        text-align:center;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
        border: 2px solid rgba(0,0,0,0.1);
    ">
        <h1 style="font-size:50px;">{predicted_icon}</h1>
        <h2 style="color:#1B4F72; margin:5px;">Prediction: {predicted_class}</h2>
        <p style="font-size:18px; margin:5px;"><strong>Confidence:</strong> {confidence*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
