import json
import joblib
import os
import time
import streamlit as st
import pandas as pd

MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "z_scaler.pkl"
FEATURE_COLS_PATH = "feature_columns.json"
TEST_FILE_PATH = "test.csv"
SCALE_COLS = ["argsNum", "returnValue"]

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, str(e)

model, scaler, FEATURE_COLS = load_artifacts()

st.set_page_config(page_title="Network-style Anomaly Detector", layout="wide")

st.markdown("""
    <style>
    .scanner-text { font-family: 'Courier New', Courier, monospace; font-size: 14px; }
    .anomaly-list { background-color: #ffeded; border-radius: 5px; padding: 10px; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Live Process Security Monitor")
st.caption("Mô phỏng bắt tín hiệu tiến trình theo thời gian thực (Wireshark Style)")

if "anomaly_history" not in st.session_state:
    st.session_state.anomaly_history = []

# Sidebar điều khiển
with st.sidebar:
    st.header("🎮 Điều khiển")
    speed = st.slider("Tốc độ quét (giây/tiến trình)", 0.1, 2.0, 0.5)
    run_btn = st.button("🚀 Bắt đầu giám sát")
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.anomaly_history = []
        st.rerun()
    
    st.divider()
    uploaded_file = st.file_uploader("Hoặc tải file test mới", type=["csv"])

source_file = uploaded_file if uploaded_file else (TEST_FILE_PATH if os.path.exists(TEST_FILE_PATH) else None)

if source_file:
    df_test = pd.read_csv(source_file)

    col_live, col_anomaly = st.columns([1.5, 1])

    with col_live:
        st.subheader("📡 Live Process Stream")
        stream_placeholder = st.empty()

    with col_anomaly:
        st.subheader("🚨 Detected Anomalies")
        anomaly_placeholder = st.empty() 

    if run_btn:
        logs = []
        for index, row in df_test.iterrows():
            raw_features = row[FEATURE_COLS].to_dict()

            X_df = pd.DataFrame([raw_features])
            X_scaled = X_df.copy()
            X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])
            
            pred = model.predict(X_scaled[FEATURE_COLS])[0]

            status_icon = "⚪" if pred == 0 else "🔴"
            log_entry = f"{status_icon} ID: {index} | Name: {row.get('name', 'Proc_'+str(index))} | Threads: {row.get('threadId', 0)}"
            logs.insert(0, log_entry)
            stream_placeholder.code("\n".join(logs[:15]))

            if pred == 1:
                anomaly_item = {
                    "id": index,
                    "name": row.get('name', f"Process_{index}"),
                    "raw_data": raw_features
                }
                st.session_state.anomaly_history.append(anomaly_item)

            with anomaly_placeholder.container():
                for item in reversed(st.session_state.anomaly_history):
                    with st.expander(f"🔴 ID: {item['id']} - {item['name']}"):
                        st.write("**Chỉ số thô (Raw Features):**")
                        st.json(item['raw_data'])
            
            time.sleep(speed)
        
        st.success("🏁 Hoàn thành quét file test.")

    elif not st.session_state.anomaly_history:
        stream_placeholder.info("Nhấn 'Bắt đầu giám sát' để chạy dữ liệu.")
else:
    st.error("❌ Không tìm thấy file test.csv. Vui lòng kiểm tra lại trong Git hoặc tải lên thủ công.")
