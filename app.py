import json
import joblib
import psutil
import os
import streamlit as st
import pandas as pd

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

model, scaler, result = load_artifacts()

if model is None:
    st.error(f"❌ Lỗi khởi tạo ứng dụng: {result}")
    st.stop()
else:
    FEATURE_COLS = result

def process_prediction(feature_dict):
    X = pd.DataFrame([feature_dict])
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLS]
  
    X_scaled = X.copy()
    X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])
  
    pred = int(model.predict(X_scaled)[0])
    prob = None
    try:
        prob = model.predict_proba(X_scaled)[0][1]
    except:
        pass
    return pred, prob, X_scaled

@st.cache_data(ttl=5)
def get_process_df():
    rows = []
    try:
        for p in psutil.process_iter(['pid', 'name', 'ppid', 'uids', 'num_threads', 'cmdline']):
            try:
                info = p.info
                rows.append({
                    "pid": info['pid'],
                    "name": info['name'] or "Unknown",
                    "parentProcessId": info['ppid'] or 0,
                    "userId": info['uids'].real if info['uids'] else 0,
                    "threadId": info['num_threads'] or 0,
                    "argsNum": len(info['cmdline']) if info['cmdline'] else 0,
                    "mountNamespace": os.getpid(),
                    "returnValue": 0
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        st.sidebar.error(f"Lỗi truy cập hệ thống: {e}")
        return pd.DataFrame()
    return pd.DataFrame(rows)

st.set_page_config(page_title="Anomaly Detection", layout="wide", page_icon="🛡️")

st.title("🛡️ OS Process Anomaly Detection")
st.caption("Phát hiện hành vi bất thường của tiến trình hệ thống")

st.sidebar.header("⚙️ Cấu hình")
mode = st.sidebar.radio("Chế độ nhập dữ liệu", ["🖥️ Chọn tiến trình đang chạy", "✍️ Nhập thủ công"])
if st.sidebar.button("🔄 Làm mới danh sách"):
    st.cache_data.clear()
    st.rerun()

X_model_input = None
current_process_name = ""

if mode == "🖥️ Chọn tiến trình đang chạy":
    st.header("🔍 Quét tiến trình hệ thống")
    df_proc = get_process_df()

    if df_proc.empty:
        st.warning("⚠️ Không thể lấy danh sách tiến trình. Vui lòng sử dụng chế độ 'Nhập thủ công'.")
    else:
        # Tạo nhãn: "facebook.exe (PID: 1234)"
        df_proc["label"] = df_proc.apply(lambda r: f"{r['name']} (PID: {r.pid})", axis=1)
        selected_label = st.selectbox("Chọn tiến trình cần kiểm tra:", df_proc["label"].tolist())
        

        row = df_proc[df_proc["label"] == selected_label].iloc[0]
        current_process_name = row['name'] 
        st.success(f"🎯 **Đang phân tích tiến trình:** `{current_process_name}`")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("PID", row.pid)
        col_info2.metric("PPID", row.parentProcessId)
        col_info3.metric("Threads", row.threadId)

        feature_dict = {
            "parentProcessId": row.parentProcessId, "userId": row.userId,
            "threadId": row.threadId, "argsNum": row.argsNum,
            "mountNamespace": row.mountNamespace, "returnValue": row.returnValue
        }
        _, _, X_model_input = process_prediction(feature_dict)
        final_feature_dict = feature_dict

else:
    st.header("✍️ Nhập thông số thủ công")
    current_process_name = "Manual Input"
    c1, c2, c3 = st.columns(3)
    p_id = c1.number_input("parentProcessId", 0)
    u_id = c1.number_input("userId", 0)
    t_id = c2.number_input("threadId", 1)
    a_num = c2.number_input("argsNum", 0)
    m_ns = c3.number_input("mountNamespace", 0)
    r_val = c3.number_input("returnValue", 0)
    
    final_feature_dict = {
        "parentProcessId": p_id, "userId": u_id, "threadId": t_id,
        "argsNum": a_num, "mountNamespace": m_ns, "returnValue": r_val
    }
    _, _, X_model_input = process_prediction(final_feature_dict)


if X_model_input is not None:
    st.divider()
    st.subheader(f"🤖 Kết quả phân tích: {current_process_name}")
    
    pred_label, prob, _ = process_prediction(final_feature_dict)
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if pred_label == 1:
            st.error(f"🚨 **PHÁT HIỆN BẤT THƯỜNG TRÊN {current_process_name.upper()}**")
            st.warning("Hành vi này có dấu hiệu xâm nhập hoặc tiến trình lạ.")
        else:
            st.success(f"✅ **{current_process_name} HOẠT ĐỘNG BÌNH THƯỜNG**")
            st.info("Không phát hiện dấu hiệu đe dọa.")
        
        if prob is not None:
            st.write(f"**Độ tin cậy:** `{prob:.2%}`")

    with col_res2:
        with st.expander("Xem chi tiết Vector đặc trưng (Scaled)"):
            st.dataframe(X_model_input)

