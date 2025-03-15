import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

#load Dataset
@st.cache_data
def load_data():
    file_path = 'factype2.xlsx'
    return pd.read_excel(file_path)

df = load_data()

#จัดการค่าที่ขาด
@st.cache_data
def impute_data(df):
    imputer = SimpleImputer(strategy="most_frequent")
    df[["เงินทุนรวม", "แรงม้า"]] = imputer.fit_transform(df[["เงินทุนรวม", "แรงม้า"]])
    return df

df = impute_data(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["เงินทุนรวม", "แรงม้า"]])

st.title("Machine Learning Model (KNN)")

technique = st.radio("Select Algorithm", ("K-Means", "DBSCAN"))

if technique == "K-Means":
    k_value = st.slider("Select K (Cluster) :K-Means", min_value=2, max_value=6, value=3)
elif technique == "DBSCAN":
    eps_value = st.slider("Select radius :DBSCAN", 0.1, 3.0, 1.0)
    min_samples_value = st.slider("Min Samples :DBSCAN", 1, 8, 4)
# elif technique == "K-NN":
#     k_nn_value = st.slider("Select neighbor K :K-NN", min_value=2, max_value=4, value=3)

#K-Means
if technique == "K-Means":
    kmeans = KMeans(n_clusters=k_value, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    with st.container():
        df["KMeans_Cluster"] = kmeans_labels

        #K-Means Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["เงินทุนรวม"], y=df["แรงม้า"], hue=df["KMeans_Cluster"], palette="viridis", ax=ax)
        ax.set_xlabel("Total capital")
        ax.set_ylabel("horsepower")
        st.pyplot(fig)

        #range K-Means
        for i in range(k_value):
            cluster_data = df[df["KMeans_Cluster"] == i]
            min_funding = cluster_data["เงินทุนรวม"].min()
            max_funding = cluster_data["เงินทุนรวม"].max()
            min_hp = cluster_data["แรงม้า"].min()
            max_hp = cluster_data["แรงม้า"].max()

            st.write(f"**Cluster (Group) {i + 1}:**")
            if min_funding == max_funding:
                st.write(f" - **Total capital:** {min_funding} millions Bath  (low eqaul high)")
            else:
                st.write(f" - **Total capital:** {min_funding} to {max_funding} millions Bath ")
            
            if min_hp == max_hp:
                st.write(f" - **Production capacity:** {min_hp} horsepower (low eqaul high)")
            else:
                st.write(f" - **Production capacity:** {min_hp} to {max_hp} horsepower")
            
            st.write("---")

#DBSCAN
if technique == "DBSCAN":
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan_labels = dbscan.fit_predict(scaled_data)

    with st.container():
        df["DBSCAN_Cluster"] = dbscan_labels

        #Outliers
        outliers_dbscan = df[df["DBSCAN_Cluster"] == -1]
        if len(outliers_dbscan) > 0:
            st.error(f"Found Outlier : {len(outliers_dbscan)} point!")
        else:
            st.success("Not found outliers in data")

        #DBSCAN Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["เงินทุนรวม"], y=df["แรงม้า"], hue=df["DBSCAN_Cluster"], palette="coolwarm", ax=ax)
        ax.set_xlabel("Total capital")
        ax.set_ylabel("horsepower")
        st.pyplot(fig)

        
        st.write("---")

