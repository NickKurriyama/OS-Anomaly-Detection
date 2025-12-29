import json
import joblib
import os
import streamlit as st
import pandas as pd

# =====================================================
# 1. CẤU HÌNH & TẢI MODEL
# =====================================================
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "z_scaler.pkl"
FEATURE_COLS_PATH = "feature_columns.json"
SCALE_COLS = ["argsNum", "returnValue"]

@st.cache_resource
def load_artifacts():
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None, f"Không tìm thấy file {MODEL_PATH}"
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, str(e)

model, scaler, FEATURE_COLS = load_artifacts()

# =====================================================
# 2. HÀM XỬ LÝ LỌC TIẾN TRÌNH (VÒNG LẶP)
# =====================================================
def scan_processes(df_input):
    """
    Duyệt qua từng dòng trong file test, dự đoán và lọc ra danh sách lỗi
    """
    anomalies = []
    normal_count = 0
    
    # Đảm bảo dataframe có đủ các cột cần thiết, thiếu thì bù bằng 0
    for col in FEATURE_COLS:
        if col not in df_input.columns:
            df_input[col] = 0

    # Lấy dữ liệu theo đúng thứ tự Feature model yêu cầu
    X_raw = df_input[FEATURE_COLS].copy()
    
    # Scale dữ liệu hàng loạt để tăng tốc độ (thay vì lặp từng dòng để scale)
    X_scaled = X_raw.copy()
    X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])
    
    # Dự đoán toàn bộ
    predictions = model.predict(X_scaled)
    
    # Nếu model có predict_proba thì lấy xác suất
    probs = [None] * len(predictions)
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except:
        pass

    # Vòng lặp duyệt qua kết quả để phân loại
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # Lưu lại thông tin tiến trình bị lỗi
            # Giả sử file test có cột 'name' hoặc 'pid', nếu không có sẽ báo 'Unknown'
            proc_info = {
                "Tên": df_input.iloc[i].get("name", "Unknown"),
                "PID": df_input.iloc[i].get("pid", "N/A"),
                "Mức độ rủi ro": f"{probs[i]:.2%}" if probs[i] is not None else "N/A"
            }
            # Thêm các chỉ số đặc trưng vào để xem lý do lỗi
            for col in FEATURE_COLS:
                proc_info[col] = df_input.iloc[i][col]
            
            anomalies.append(proc_info)
        else:
            normal_count += 1
            
    return pd.DataFrame(anomalies), normal_count

# =====================================================
# 3. GIAO DIỆN CHÍNH
# =====================================================
st.set_page_config(page_title="Batch Anomaly Detector", layout="wide", page_icon="🛡️")

st.title("🛡️ Batch Process Security Scanner")
st.caption("Tải lên file dữ liệu test để mô hình tự động quét và lọc tiến trình độc hại")

# Sidebar: Hướng dẫn file mẫu
with st.sidebar:
    st.header("📂 Hướng dẫn file Test")
    st.write("File cần có các cột:")
    st.code(", ".join(FEATURE_COLS))
    st.info("Hệ thống sẽ lặp qua từng tiến trình để phân tích.")

# Giao diện tải File
uploaded_file = st.file_uploader("Chọn file dữ liệu (CSV hoặc Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu
    try:
        if uploaded_file.name.endswith('.csv'):
            df_test = pd.read_csv(uploaded_file)
        else:
            df_test = pd.read_excel(uploaded_file)
        
        st.write(f"📊 Đã tải lên **{len(df_test)}** tiến trình.")
        
        if st.button("🚀 Bắt đầu quét hệ thống"):
            with st.spinner('Đang chạy vòng lặp kiểm tra từng tiến trình...'):
                df_anomalies, normal_count = scan_processes(df_test)
            
            # Hiển thị kết quả tổng quan bằng cột
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số quét", len(df_test))
            c2.metric("Tiến trình Bình thường", normal_count)
            c3.metric("Tiến trình Bất thường", len(df_anomalies), delta_color="inverse")

            # Hiển thị danh sách bị lỗi
            if not df_anomalies.empty:
                st.error(f"🚨 Phát hiện {len(df_anomalies)} tiến trình có dấu hiệu nguy hiểm!")
                st.subheader("📋 Danh sách đen (Blacklist) đã lọc:")
                
                # Highlight các dòng lỗi
                st.dataframe(df_anomalies.style.background_gradient(cmap='Reds', subset=['Mức độ rủi ro'] if "Mức độ rủi ro" in df_anomalies.columns else []))
                
                # Cho phép tải về kết quả lỗi
                csv = df_anomalies.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải danh sách lỗi (.csv)", csv, "detected_anomalies.csv", "text/csv")
            else:
                st.success("✅ Tuyệt vời! Không phát hiện tiến trình nào bất thường trong file này.")

    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")

else:
    # Giao diện khi chưa tải file
    st.info("Vui lòng tải lên file dữ liệu test để bắt đầu quá trình lọc.")
    # Hiển thị ví dụ cấu trúc dữ liệu model cần
    st.subheader("Ví dụ cấu trúc dữ liệu hợp lệ:")
    example_data = pd.DataFrame([[0, 0, 1, 0, 0, 0]], columns=FEATURE_COLS)
    st.table(example_data)
