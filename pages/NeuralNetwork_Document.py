import streamlit as st

st.title(":blue[NeuralNetwork Explanation]")

st.markdown("""### Dataset :green[Kaggle]""")
st.markdown("""#### แหล่งข้อมูล (Reference)
            https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data""")
st.markdown("""เราจะใช้ Dataset นี้สำหรับงาน Image Processing และการตรวจสอบรูปภาพ โดยโมเดลที่ใช้จะแยกประเภทของภาพออกเป็น 6 หมวดหมู่ ได้แก่... """)
code = '''
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']'''
st.code(code, language="python")

st.markdown("""
โดยมี <span style="color:green"><b>ข้อดี</b></span> และ <span style="color:red"><b>ข้อเสีย</b></span> ดังนี้:

- <span style="color:green"><b>ข้อดี</b></span>: โมเดลสามารถเรียนรู้ได้อย่างรวดเร็วและให้ความแม่นยำสูงในการจำแนกประเภทของข้อมูลที่มีอยู่ในชุดฝึก  
- <span style="color:red"><b>ข้อเสีย</b></span>: หากป้อนข้อมูลที่โมเดลไม่เคยเรียนรู้มาก่อน (Out-of-distribution data) อาจส่งผลให้เกิดการคาดการณ์ที่ไม่ถูกต้อง  
""", unsafe_allow_html=True)



st.text("ตัวอย่างโค้ดในการ import Intel Dataset")
code2 = '''
#Kaggle API
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#Download Dataset
!kaggle datasets download -d puneet6060/intel-image-classification

'''
st.code(code2, language="python")

refIMG = "https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data"

col1, col2 = st.columns(2)
with col1 :
    st.image("intel_data/seg_test/seg_test/street/24052.jpg",caption=refIMG, width=350)
with col2 :
    st.image("intel_data/seg_test/seg_test/mountain/24066.jpg",caption=refIMG, width=350)

st.markdown(""" ### Model ที่ใช้ :blue[Convolutional Neural Network (CNN)]""")

st.markdown(""" #### อธิบายโครงสร้าง CNN Model""")

st.text("Layers :")

col1, col2 = st.columns(2)
with col1 :
    st.image("image/LayerDetail.png")
with col2 :
    st.image("image/LayerCol.png")

st.markdown("""#### ส่วนของการเตรียมข้อมูล (Data Preprocessing)
    ใช้ ImageDataGenerator ปรับค่า Pixel (0-255 → 0-1)
    แบ่งข้อมูล 90% Train / 10% Validation       
    ปรับขนาดภาพเป็น 150x150 px""")

st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# เตรียมชุดข้อมูล Train และ Validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="categorical", subset="training"
)
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="categorical", subset="validation"
)

# เตรียมชุดข้อมูล Test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode="categorical", shuffle=False
)
""", language="python")

st.markdown("""#### การสร้างและคอมไพล์โมเดล (Model Architecture & Compilation)""")

st.markdown("""
โมเดลที่ใช้คือ **Convolutional Neural Network (CNN)** สำหรับการจำแนกภาพ 6 หมวดหมู่  
โดยใช้ **3 ชั้นของ Convolutional Layers + MaxPooling** เพื่อดึงคุณสมบัติของภาพ  
จากนั้น Flatten และเชื่อมต่อกับ Fully Connected Layers พร้อม Dropout เพื่อลด Overfitting  

### **🔹 โครงสร้างของโมเดล**
- **Conv2D + ReLU** → ดึง Feature จากภาพ
- **BatchNormalization** → ปรับค่าให้เสถียร
- **MaxPooling2D** → ลดขนาดของภาพ
- **Flatten + Dense Layers** → แปลงเป็นเวกเตอร์สำหรับจำแนกคลาส
- **Softmax Output (6 Classes)** → จำแนกภาพเป็น 6 หมวดหมู่

โมเดลถูกคอมไพล์ด้วย **Adam Optimizer** และใช้ **Categorical Crossentropy** เป็น Loss Function  
พร้อมแสดงผลลัพธ์ด้วย **Accuracy Metric**  
""")

st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# สร้างโมเดล CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    
    Dense(6, activation="softmax")
])

# คอมไพล์โมเดล
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# แสดงสรุปโมเดล
model.summary()
""", language="python")

st.markdown("""#### การฝึกโมเดล (Model Training)""")

st.markdown("""
โมเดลถูกฝึกโดยใช้ **Train Set** และตรวจสอบผลลัพธ์ด้วย **Validation Set**  
โดยใช้ **EarlyStopping** เพื่อหยุดการฝึกหาก `val_loss` ไม่ลดลง และ **ReduceLROnPlateau**  
เพื่อปรับ Learning Rate อัตโนมัติเมื่อโมเดลเริ่ม Overfitting  

#### **🔹 การตั้งค่าการฝึก**
- **Epochs:** 30 รอบ (หรือหยุดก่อนหาก `val_loss` ไม่ลดลง)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Callbacks:** EarlyStopping & ReduceLROnPlateau

ผลลัพธ์จากการฝึกสามารถนำไปใช้วิเคราะห์โมเดลและปรับปรุงในขั้นตอนถัดไป  
""")

st.code("""
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ตั้งค่า Callback เพื่อลด Overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# ฝึกโมเดล
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)
""", language="python")

st.markdown("""#### การประเมินผลโมเดล (Model Evaluation)""")
st.code("""
    test_loss, test_acc = model.evaluate(test_generator)
""", language="python")

st.image("image/evaluateModel.png")