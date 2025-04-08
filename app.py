import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดล
@st.cache_resource
def load_models():
    imputer = joblib.load('imputer_model.pkl')
    scaler = joblib.load('scaler_model.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    return imputer, scaler, kmeans

imputer, scaler, kmeans = load_models()

st.title("🎯 K-Means Clustering Prediction App")
st.write("อัปโหลดไฟล์ CSV ข้อมูลใหม่ แล้วระบบจะทำนายคลัสเตอร์ให้อัตโนมัติ")

uploaded_file = st.file_uploader("📂 เลือกไฟล์ CSV", type=["csv"])

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)

        # ตรวจสอบว่ามีอย่างน้อย 35 คอลัมน์
        if new_data.shape[1] < 35:
            st.error("ไฟล์ CSV ต้องมีอย่างน้อย 35 คอลัมน์แรกเพื่อให้ใช้กับโมเดลนี้ได้")
        else:
            X_new = new_data.iloc[:, :35]
            X_new_imputed = imputer.transform(X_new)
            X_new_scaled = scaler.transform(X_new_imputed)
            new_clusters = kmeans.predict(X_new_scaled)

            new_data['Cluster'] = new_clusters

            st.success("🎉 ทำนายคลัสเตอร์สำเร็จแล้ว!")
            st.write("📊 จำนวนในแต่ละคลัสเตอร์:")
            st.dataframe(new_data['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

            with st.expander("🔍 ดูข้อมูลทั้งหมด"):
                st.dataframe(new_data)

            csv_download = new_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 ดาวน์โหลดไฟล์พร้อมคลัสเตอร์", data=csv_download, file_name="clustered_data.csv", mime='text/csv')

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
