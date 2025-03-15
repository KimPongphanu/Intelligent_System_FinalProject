import streamlit as st
import os
import gdown

MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/file/d/1KUSIQCJPyBK4ndwdMiwpLWBsiDAAM-7c/view?usp=sharing"  

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading CNN model... (please wait)")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")


st.markdown('<div class="custom-title">MACHINE LEARNING</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Machine_Learning_Document"):
        st.switch_page("pages/Machine_Learning_Document.py")

with col2:
    if st.button("Machine_Learning_Model"):
        st.switch_page("pages/Machine_Learning_Model.py")

st.markdown("---")

st.markdown('<div class="custom-title">NEURAL NETWORK</div>', unsafe_allow_html=True)
col3, col4 = st.columns([1, 1])

with col3:
    if st.button("NeuralNetwork_Document"):
        st.switch_page("pages/NeuralNetwork_Document.py")

with col4:
    if st.button("NeuralNetwork_Model"):
        st.switch_page("pages/NeuralNetwork_Model.py")


st.markdown("""### สามารถ ดู GitHub ได้ กดที่นี่
#### **Link**: [Click here](https://github.com/KimPongphanu/IS_Project.git)  
""")

st.markdown("""### สามารถ โหลด model CNN เพื่อทดลองได้ ได้ กดที่นี่
#### **Link**: [Click here](https://drive.google.com/file/d/1KUSIQCJPyBK4ndwdMiwpLWBsiDAAM-7c/view?usp=sharing)  
""")