import json
import joblib
import psutil
import os
import streamlit as st
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "z_scaler.pkl"
FEATURE_COLS_PATH = "feature_columns.json"
SCALE_COLS = ["argsNum", "returnValue"]

# =====================================================
# LOAD ARTIFACTS (Thêm try-except để chống trắng trang)
# =====================================================
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

# Khởi tạo load
model, scaler, result = load_artifacts()

# Kiểm tra nếu load lỗi thì dừng app và hiện lỗi ngay
if model is None:
    st.error(f"❌ Lỗi khởi tạo ứng dụng: {result}")
    st.info("Hãy đảm bảo các file .pkl và .json đã được upload lên GitHub cùng thư mục với file code.")
    st.stop()
else:
    FEATURE_COLS = result

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="OS Process Anomaly Detection",
    layout="wide",
    page_icon="🛡️"
)

st.title("🛡️ Real-time Process Anomaly Detection")
st.caption("Ứng dụng phát hiện tiến trình bất thường dựa trên Machine Learning")

# =====================================================
# UTILS
# =====================================================
def build_model_input(feature_dict: dict):
    X = pd.DataFrame([feature_dict])
    # Đảm bảo đủ feature theo đúng thứ tự model yêu cầu
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLS]
    
    # Scale dữ liệu
    X_scaled = X.copy()
    try:
        X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])
    except Exception as e:
        st.warning(f"Lỗi khi scale dữ liệu: {e}")
    return X_scaled

@st.cache_data(ttl=5)
def get_process_df():
    rows = []
    # Streamlit Cloud chạy trên Linux container, psutil có thể bị hạn chế
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
                    "mountNamespace": os.getpid(), # Giá trị giả định
                    "returnValue": 0
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        st.sidebar.error(f"Cảnh báo hệ thống: {e}")
        # Trả về dữ liệu trống nếu bị hệ thống chặn hoàn toàn
        return pd.DataFrame()

    return pd.DataFrame(rows)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Cấu hình")
mode = st.sidebar.radio(
    "Chọn phương thức nhập dữ liệu",
    ["✍️ Manual Input (Khuyên dùng trên Cloud)", "🖥️ Select Running Process"]
)

refresh = st.sidebar.button("🔄 Làm mới danh sách tiến trình")
if refresh:
    get_process_df.clear()

# Biến trung gian để chứa input cho model
X_model = None

# =====================================================
# MODE 1 — SELECT RUNNING PROCESS
# =====================================================
if mode == "🖥️ Select Running Process":
    st.header("🖥️ Kiểm tra tiến trình đang chạy")
    df_proc = get_process_df()

    if df_proc.empty:
        st.warning("⚠️ Không thể quét danh sách tiến trình (Có thể do hạn chế quyền trên Cloud). Hãy sử dụng Manual Input.")
    else:
        df_proc["label"] = df_proc.apply(lambda r: f"PID {r.pid} — {r.name}", axis=1)
        selected_label = st.selectbox("Chọn một tiến trình từ danh sách", df_proc["label"].tolist())
        row = df_proc[df_proc["label"] == selected_label].iloc[0]

        # Hiển thị thông tin Raw
        st.subheader("📊 Thông tin hệ thống (Raw)")
        st.json({
            "PID": int(row.pid), "Name": row.name, "PPID": int(row.parentProcessId),
            "User ID": int(row.userId), "Threads": int(row.threadId), "ArgsNum": int(row.argsNum)
        })

        feature_dict = {
            "parentProcessId": row.parentProcessId, "userId": row.userId,
            "threadId": row.threadId, "argsNum": row.argsNum,
            "mountNamespace": row.mountNamespace, "returnValue": row.returnValue
        }
        X_model = build_model_input(feature_dict)

# =====================================================
# MODE 2 — MANUAL INPUT
# =====================================================
else:
    st.header("✍️ Nhập thông số thủ công")
    col1, col2, col3 = st.columns(3)
    with col1:
        parent_pid = st.number_input("parentProcessId", value=0)
        user_id = st.number_input("userId", value=0)
    with col2:
        thread_id = st.number_input("threadId", value=1)
        args_num = st.number_input("argsNum", value=0)
    with col3:
        mount_ns = st.number_input("mountNamespace", value=0)
        return_value = st.number_input("returnValue", value=0)

    feature_dict = {
        "parentProcessId": parent_pid, "userId": user_id, "threadId": thread_id,
        "argsNum": args_num, "mountNamespace": mount_ns, "returnValue": return_value
    }
    X_model = build_model_input(feature_dict)

# =====================================================
# INFERENCE (Dự đoán)
# =====================================================
if X_model is not None:
    st.divider()
    st.subheader("🧬 Vector đặc trưng (Model Input)")
    st.dataframe(X_model)

    st.subheader("🤖 Kết quả dự đoán từ Model")
    try:
        # Dự đoán
        prediction = model.predict(X_model)
        pred_label = int(prediction[0])
        
        # Thử lấy xác suất nếu model hỗ trợ
        try:
            prob = model.predict_proba(X_model)[0][1]
            st.write(f"Độ tin cậy của bất thường: {prob:.2%}")
        except:
            pass

        if pred_label == 1:
            st.error("🚨 PHÁT HIỆN BẤT THƯỜNG (Anomaly Detected)")
            st.warning("Tiến trình này có các dấu hiệu không giống với hoạt động bình thường của hệ thống.")
        else:
            st.success("✅ TIẾN TRÌNH BÌNH THƯỜNG (Normal Process)")
            st.info("Không phát hiện dấu hiệu xâm nhập hoặc lỗi hệ thống.")

    except Exception as e:
        st.exception(f"Lỗi khi thực hiện dự đoán: {e}")
