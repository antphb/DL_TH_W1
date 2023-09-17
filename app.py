import streamlit as st
import joblib
import numpy as np
import pandas as pd

def prediction(data):
    model=joblib.load('xgboost.pkl')
    lable=model.predict(np.array([data]))[0]
    return str(lable)

def prediction_file(path_file):
    model=joblib.load('xgboost.pkl')
    data=pd.read_csv(path_file,encoding="utf-8")
    lable=model.predict(data)
    data["lable"]=lable
    return data


# Tiêu đề ứng dụng
st.title("20003005 - Nguyễn Đình Thanh")

menu=st.sidebar.selectbox("Chọn chức năng",['Nhập dữ liệu','Upload file'])

lables=[0,1,2]
# show list lables
st.write("Danh sách các nhãn:", lables)

if menu == "Nhập dữ liệu":
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        selected_value1 = st.slider("Cột 1", 0.0, 10.0, step=0.1)
    with col2:
        selected_value2 = st.slider("Cột 2", 0.0, 10.0, step=0.1)
    with col3:
        selected_value3 = st.slider("Cột 3", 0.0, 10.0, step=0.1)
    with col4:
        selected_value4 = st.slider("Cột 4", 0.0, 10.0, step=0.1)
    with col5:
        selected_value5 = st.slider("Cột 5", 0.0, 10.0, step=0.1)
    with col6:
        selected_value6 = st.slider("Cột 6", 0.0, 10.0, step=0.1)
    with col7:
        selected_value7 = st.slider("Cột 7", 0.0, 10.0, step=0.1)
    if st.button("Dự đoán"):
        data=[selected_value1,selected_value2,selected_value3,selected_value4,selected_value5,selected_value6,selected_value7]
        lables_predict=prediction(data)
        st.markdown("<p>Dự đoán: <span style='color: red'>"+lables_predict+"</span></p>", unsafe_allow_html=True)
else:
    file=st.file_uploader("Upload file csv",type=["csv"])
    # checkbox file văn bản demo
    if st.checkbox("File demo") and file is None:
        path_file="file_demo.csv"
        with open(path_file,encoding="utf-8") as f:
            text=f.read()
        st.text_area("File demo",text,height=300,disabled=True)
        if st.button("Dự đoán"):
            lables_predict=prediction_file(path_file)
            # print(lables_predict)
            st.dataframe(lables_predict)
    elif file is not None:
        print(file.name)
        # display none checkbox file demo
        text=file.read().decode("utf-8")
        st.text_area("Văn bản",text,height=300,disabled=True)
        if st.button("Dự đoán"):
            lables_predict=prediction_file(file.name)
            # print(lables_predict)
            st.dataframe(lables_predict)