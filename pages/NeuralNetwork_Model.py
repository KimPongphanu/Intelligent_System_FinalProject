import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os
import gdown

MODEL_PATH = "intel_cnn_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1KUSIQCJPyBK4ndwdMiwpLWBsiDAAM-7c" 

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading CNN model... (please wait)")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if file_size < 1:
        st.error(f"❌ Model file is corrupted (Size: {file_size:.2f} MB). Please check Google Drive link.")
        st.stop()

    st.success(f"✅ Model downloaded successfully! (Size: {file_size:.2f} MB)")

st.info("🔄 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop() 

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ฟังก์ชันเตรียมภาพก่อนนำไปทำนาย
def prepare_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# ฟังก์ชันสุ่มภาพจากชุดข้อมูล Test
def random_intel_image():
    categories = os.listdir("intel_data/seg_test/seg_test")  
    selected_category = random.choice(categories)
    category_path = os.path.join("intel_data/seg_test/seg_test", selected_category)
    image_name = random.choice(os.listdir(category_path))
    image_path = os.path.join(category_path, image_name)
    
    image = Image.open(image_path)
    return image, selected_category

# ส่วนแสดงผล Streamlit
st.title("CNN Model")

st.markdown("""
#### Features ที่น่าสนใจ มี 2 อย่างให้ทดลองใช้งาน คือ
1. อัพโหลดรูปภาพจากอุปกรณ์ของผู้ใช้งาน
2. สุ่มรูปภาพ เพื่อนำมาทดลองใช้งาน  
""")

# ฟังก์ชันการอัปโหลดภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    image = prepare_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.write(f"Prediction: {classes[predicted_class]}")
    st.write(f"Confidence: {np.max(prediction):.2f}")

# ฟังก์ชันสุ่มภาพจาก Dataset
if st.button("สุ่มภาพตัวอย่าง", type="primary"):
    if uploaded_file is None:
        image, actual_class = random_intel_image()
        st.image(image, caption=f"Random Image - {actual_class}", width=300)

        image = prepare_image(image)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        st.write(f"Prediction: {classes[predicted_class]}")
        st.write(f"Actual class: {actual_class}")
        st.write(f"Confidence: {np.max(prediction):.2f}")
