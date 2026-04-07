import streamlit as st
import pickle
import numpy as np

# Load model
with open('model_linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Prediksi Indeks Performa Akademik Mahasiswa")

st.write("Masukkan data kebiasaan belajar mahasiswa")

sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
hours_studied = st.number_input("Hours Studied", 0.0, 12.0, 2.0)
previous_score = st.number_input("Previous Score", 0.0, 100.0, 75.0)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ("Yes", "No")
)

sample_papers = st.number_input(
    "Sample Question Papers Practiced",
    0, 20, 2
)

if st.button("Predict"):

    if extracurricular == "Yes":
        extracurricular = 1
    else:
        extracurricular = 0

    data = np.array([[hours_studied,
                      previous_score,
                      extracurricular,
                      sleep_hours,
                      sample_papers]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"Predicted Performance Index: {prediction[0]:.2f}")