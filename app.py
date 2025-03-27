import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 모델 및 스케일러 로드
model = load_model("model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

st.title("🧠 Tacrolimus 농도 예측기")
st.markdown("Frequency, Real, -Imag, Phase 값을 입력하면 예측된 농도를 출력합니다.")

# 입력값
freq = st.number_input("📡 Frequency (3000 이하)", value=1000.0)
real = st.number_input("🧪 Real", value=1200.0)
neg_imag = st.number_input("📉 -Imag", value=500.0)
phase = st.number_input("🌀 Phase", value=-45.0)

# 예측 버튼
if st.button("예측하기"):
    try:
        input_data = np.array([[freq, real, neg_imag, phase]])
        X_scaled = scaler_X.transform(input_data)
        y_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)
        st.success(f"✅ 예측된 농도: {y_pred[0][0]:.2f}")
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
