import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# ---------------------------
# Load Model
# ---------------------------
model = tf.keras.models.load_model("plant_model.h5")

# ---------------------------
# Class Labels (EDIT if needed)
# ---------------------------
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Pepper_Bacterial_spot",
    "Pepper_healthy"
]

# ---------------------------
# Remedies
# ---------------------------
solutions = {
    "Tomato_Early_blight": "Remove infected leaves and apply fungicide.",
    "Tomato_Late_blight": "Use copper fungicide and avoid water on leaves.",
    "Tomato_healthy": "Plant is healthy. Maintain proper care.",

    "Potato_Early_blight": "Use fungicide and remove affected leaves.",
    "Potato_Late_blight": "Improve air circulation and apply fungicide.",
    "Potato_healthy": "Plant is healthy.",

    "Pepper_Bacterial_spot": "Use copper spray and remove infected leaves.",
    "Pepper_healthy": "Plant is healthy."
}

# ---------------------------
# Functions
# ---------------------------
def get_plant_name(label):
    return label.split("_")[0]

def check_health(label):
    return "Healthy" if "healthy" in label.lower() else "Affected"

def get_severity(conf):
    if conf > 0.9:
        return "🔴 Severe"
    elif conf > 0.7:
        return "🟡 Moderate"
    else:
        return "🟢 Mild"

# ---------------------------
# Weather API
# ---------------------------
def get_weather(city):
    API_KEY = "4e372309042115949b81e5b80359cfe5"  # 🔥 Replace this
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        data = requests.get(url).json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temp, humidity
    except:
        return None, None

def weather_advice(temp, humidity):
    if temp is None:
        return "Weather data not available"
    if humidity > 70:
        return "⚠️ High humidity → fungal risk"
    elif temp > 35:
        return "☀️ High temperature → water more"
    else:
        return "✅ Weather is suitable"

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Smart Plant Disease Detection System")

# Input options
uploaded_file = st.file_uploader("📁 Upload Leaf Image")
camera_file = st.camera_input("📸 Or Take a Photo")

# Reset button
if st.button("🔄 Reset"):
    st.rerun()

# Weather input
city = st.text_input("🌦️ Enter City for Weather")

# Select image source
img = None
if uploaded_file:
    img = Image.open(uploaded_file)
elif camera_file:
    img = Image.open(camera_file)

# ---------------------------
# Prediction
# ---------------------------
if img:
    st.image(img, caption="Leaf Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = np.max(prediction)

    plant = get_plant_name(label)
    health = check_health(label)
    severity = get_severity(confidence)
    remedy = solutions.get(label, "No solution available")

    # ---------------------------
    # Output
    # ---------------------------
    st.subheader("🌱 Plant Information")
    st.write("Plant Name:", plant)
    st.write("Disease:", label)

    st.subheader("📊 Status")

    if health == "Healthy":
        st.success("✅ Plant is Healthy")
    else:
        st.error("⚠️ Plant is Affected")

    st.write(f"Confidence: {confidence*100:.2f}%")
    st.write("Severity Level:", severity)

    st.subheader("💊 Remedy")
    st.write(remedy)

    # ---------------------------
    # Weather
    # ---------------------------
    if city:
        temp, humidity = get_weather(city)

        st.subheader("🌦️ Weather Report")
        if temp is not None:
            st.write(f"🌡️ Temperature: {temp}°C")
            st.write(f"💧 Humidity: {humidity}%")
            st.write(weather_advice(temp, humidity))
        else:
            st.warning("Could not fetch weather data")

    # ---------------------------
    # General Tips
    # ---------------------------
    st.subheader("🌿 General Plant Care Tips")
    st.markdown("""
    - 💧 Water regularly  
    - ☀️ Provide sufficient sunlight  
    - ✂️ Remove infected leaves  
    - 🌱 Use proper fertilizers  
    """)