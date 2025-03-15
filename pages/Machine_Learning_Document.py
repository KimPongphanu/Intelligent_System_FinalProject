import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data():
    file_path = "factype2.xlsx" 
    return pd.read_excel(file_path)

df = load_data()

st.markdown("""### โหลดข้อมูล (Data Loading)""")

st.markdown(""" 
ขั้นตอนนี้ช่วยให้มั่นใจว่าข้อมูลได้รับการนำเข้าอย่างถูกต้องและพร้อมสำหรับการประมวลผลต่อไป
""")

st.code("""
#load Dataset
@st.cache_data
def load_data():
    file_path = 'C:/Users/Acer/Desktop/IS_Project/factype2.xlsx'
    return pd.read_excel(file_path)

df = load_data()
""", language="python")

st.markdown("""
- **อ้างอิงแหล่งข้อมูล**: [data.go.th](https://data.go.th/dataset/factype2)  
""")

st.markdown("""
### โครงสร้างข้อมูล
- **FID**: รหัสเฉพาะของแต่ละรายการ
- **รหัสประเภท**: หมวดหมู่ของแต่ละธุรกิจ
- **จำนวน**: จำนวนของรายการในแต่ละหมวดหมู่
- **ปีข้อมูล**: ปีที่บันทึกข้อมูล
- **เงินทุนรวม**: ปริมาณเงินลงทุนของธุรกิจ (Feature หลักที่ใช้วิเคราะห์)
- **แรงม้า**: กำลังการผลิตหรือพลังงานที่ใช้ (Feature หลักที่ใช้วิเคราะห์)
""")

st.markdown("### ตัวอย่างข้อมูล")
st.dataframe(df.head())

st.markdown("""
- คุณลักษณะสำคัญ เช่น **เงินทุนรวม** และ **แรงม้า** มีบทบาทหลักในการแบ่งกลุ่มธุรกิจ
- ข้อมูลอาจมีค่าที่หายไป ซึ่งจำเป็นต้องได้รับการจัดการในขั้นตอนถัดไป
""")

st.markdown("""### การเตรียมข้อมูล (Data Preprocessing)""")

st.markdown("""
ทำความสะอาดข้อมูล ตรวจสอบค่าที่หายไป และทำให้ข้อมูลพร้อมสำหรับการสร้างโมเดล Machine Learning
""")

st.markdown("""
### การตรวจสอบค่าที่หายไป (Missing Values)
""")

missing_data = df.isnull().sum()
st.write("**Missing Values in Dataset:**")
st.dataframe(missing_data)

st.markdown("""
### จัดการกับค่าที่หายไป (Imputation)
เราจะใช้เทคนิค **Imputation** เพื่อเติมค่าลงไป โดยเลือกกลยุทธ์ที่เหมาะสม:
- **ใช้ค่าที่พบบ่อยที่สุด (Most Frequent)**
- **ใช้ค่ามัธยฐาน (Median) หรือค่าเฉลี่ย (Mean)**
- **ลบแถวที่มีค่าหายไป** หากจำนวนแถวนั้นมีจำนวนน้อยและไม่กระทบต่อการวิเคราะห์ข้อมูล
""")

@st.cache_data
def impute_data(df):
    imputer = SimpleImputer(strategy="most_frequent")
    df[["เงินทุนรวม", "แรงม้า"]] = imputer.fit_transform(df[["เงินทุนรวม", "แรงม้า"]])
    return df

df = impute_data(df)

st.markdown("### ข้อมูลหลังจากแก้ไข missing values")
st.dataframe(df.head())

st.markdown("""
### ปรับขนาดข้อมูล (Scaling)
            ใช้ **StandardScaler** เพื่อลดความแตกต่างของค่าตัวเลขและช่วยให้โมเดลเรียนรู้ได้ดีขึ้น 
""")

st.code('''scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["เงินทุนรวม", "แรงม้า"]])''',language="python")

st.markdown("""
### K-MEANS
            K-Means เป็นอัลกอริทึมที่ใช้ในการแบ่งกลุ่มข้อมูล (Clustering)
โดยเราต้องกำหนดจำนวนกลุ่ม (k) และให้โมเดลจัดกลุ่มข้อมูลที่มีความคล้ายคลึงกัน
""")

st.code('''kmeans = KMeans(n_clusters=k_value, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)''',language="python")

st.markdown("""
### DBSCAN
    เป็นอีกหนึ่งอัลกอริทึมสำหรับการแบ่งกลุ่มข้อมูล
    ซึ่งแตกต่างจาก K-Means ตรงที่ DBSCAN ไม่ต้องกำหนดจำนวนกลุ่มล่วงหน้า
    แต่ใช้วิธีวัดความหนาแน่นของข้อมูล และสามารถตรวจจับจุดข้อมูลแปลกปลอม (Outliers) ได้
""")

st.code('''dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        dbscan_labels = dbscan.fit_predict(scaled_data)''',language="python")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["เงินทุนรวม", "แรงม้า"]])

k_value = 3  
eps_value = 1.0
min_samples_value = 4  

kmeans = KMeans(n_clusters=k_value, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
df["KMeans_Cluster"] = kmeans_labels

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
dbscan_labels = dbscan.fit_predict(scaled_data)
df["DBSCAN_Cluster"] = dbscan_labels

st.markdown("""
## ตัวอย่างการใช้งานสำหรับโมเดล

ข้อมูลสำหรับการแบ่งกลุ่ม (Clustering) 
**ค่าที่ใช้:**  
- จำนวน Cluster สำหรับ K-Means: **3**  
- ค่าระยะห่าง Epsilon สำหรับ DBSCAN: **1.0**  
- จำนวนจุดขั้นต่ำในคลัสเตอร์ (Min Samples): **4**  
""")

st.markdown("## 🔹 K-Means Visualization")

fig, ax = plt.subplots()
sns.scatterplot(x=df["เงินทุนรวม"], y=df["แรงม้า"], hue=df["KMeans_Cluster"], palette="viridis", ax=ax)
ax.set_xlabel("Total Capital")
ax.set_ylabel("Horsepower")
ax.set_title("K-Means Clustering")
st.pyplot(fig)

st.markdown("""ในการใช้งานจริงอาจจะใช้แยก โรงงาน ขนาด เล็ก กลาง ใหญ่""")

for i in range(k_value):
    cluster_data = df[df["KMeans_Cluster"] == i]
    min_funding = cluster_data["เงินทุนรวม"].min()
    max_funding = cluster_data["เงินทุนรวม"].max()
    min_hp = cluster_data["แรงม้า"].min()
    max_hp = cluster_data["แรงม้า"].max()

    st.write(f"**Cluster {i + 1}:**")
    st.write(f"- **Total Capital:** {min_funding} to {max_funding} Millions Bath")
    st.write(f"- **Horsepower:** {min_hp} to {max_hp}")
    st.write("---")

st.markdown("## 🔹 DBSCAN Visualization")

fig, ax = plt.subplots()
sns.scatterplot(x=df["เงินทุนรวม"], y=df["แรงม้า"], hue=df["DBSCAN_Cluster"], palette="coolwarm", ax=ax)
ax.set_xlabel("Total Capital")
ax.set_ylabel("Horsepower")
ax.set_title("DBSCAN Clustering")
st.pyplot(fig)

st.markdown("""ในการใช้งานจริง จะใช้เพื่อหากลุ่มลูกค้าส่วนใหญ่""")

outliers_dbscan = df[df["DBSCAN_Cluster"] == -1]
if len(outliers_dbscan) > 0:
    st.error(f"⚠️ พบ Outliers จำนวน: {len(outliers_dbscan)} จุด")
else:
    st.success("✅ ไม่พบ Outliers ในข้อมูล")