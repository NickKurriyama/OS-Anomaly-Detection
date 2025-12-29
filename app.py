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
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_artifacts()

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="OS Process Anomaly Detection",
    layout="wide"
)

st.title("🛡️ Real-time Process Anomaly Detection")
st.caption("Label-based anomaly detection (0 = Normal, 1 = Anomaly)")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Mode")
mode = st.sidebar.radio(
    "Select input mode",
    ["🖥️ Select Running Process", "✍️ Manual Input"]
)

refresh = st.sidebar.button("🔄 Refresh process list")

# =====================================================
# UTILS
# =====================================================
def build_model_input(feature_dict: dict):
    """
    Build đúng input cho model:
    - đúng feature
    - đúng thứ tự
    - chỉ scale argsNum + returnValue
    """
    X = pd.DataFrame([feature_dict])

    # đảm bảo đủ feature
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0

    X = X[FEATURE_COLS]

    X_scaled = X.copy()
    X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])

    return X_scaled


@st.cache_data(ttl=3)
def get_process_df():
    rows = []

    for p in psutil.process_iter():
        try:
            rows.append({
                "pid": p.pid,
                "name": p.name(),
                "parentProcessId": p.ppid(),
                "userId": p.uids().real if p.uids() else 0,
                "threadId": p.num_threads(),
                "argsNum": len(p.cmdline()) if p.cmdline() else 0,
                "mountNamespace": os.getpid(),
                "returnValue": 0
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # fallback – KHÔNG bỏ process
            rows.append({
                "pid": p.pid,
                "name": "AccessDenied",
                "parentProcessId": 0,
                "userId": 0,
                "threadId": 0,
                "argsNum": 0,
                "mountNamespace": os.getpid(),
                "returnValue": 0
            })

    return pd.DataFrame(rows)


if refresh:
    get_process_df.clear()

# =====================================================
# MODE 1 — SELECT RUNNING PROCESS
# =====================================================
if mode == "🖥️ Select Running Process":

    st.header("🖥️ Select Running Process")

    df_proc = get_process_df()

    if df_proc.empty:
        st.warning("⚠️ No process found")
        st.stop()

    # label hiển thị
    df_proc["label"] = df_proc.apply(
        lambda r: f"PID {r.pid} — {r.name}", axis=1
    )

    selected_label = st.selectbox(
        "Select Process",
        df_proc["label"].tolist()
    )

    row = df_proc[df_proc["label"] == selected_label].iloc[0]

    # ---------- RAW INFO ----------
    st.subheader("🖥️ Raw System Information")
    st.json({
        "PID": int(row.pid),
        "Name": row.name,
        "PPID": int(row.parentProcessId),
        "User ID": int(row.userId),
        "Threads": int(row.threadId),
        "ArgsNum": int(row.argsNum)
    })

    # ---------- FEATURE VECTOR ----------
    st.subheader("🧬 Feature Vector")

    feature_dict = {
        "parentProcessId": row.parentProcessId,
        "userId": row.userId,
        "threadId": row.threadId,
        "argsNum": row.argsNum,
        "mountNamespace": row.mountNamespace,
        "returnValue": row.returnValue
    }

    X_model = build_model_input(feature_dict)
    st.dataframe(X_model)

# =====================================================
# MODE 2 — MANUAL INPUT
# =====================================================
else:
    st.header("✍️ Manual Feature Input")

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
        "parentProcessId": parent_pid,
        "userId": user_id,
        "threadId": thread_id,
        "argsNum": args_num,
        "mountNamespace": mount_ns,
        "returnValue": return_value
    }

    st.subheader("🧬 Feature Vector")
    X_model = build_model_input(feature_dict)
    st.dataframe(X_model)

# =====================================================
# INFERENCE
# =====================================================
st.divider()
st.subheader("🤖 Model Inference")

try:
    pred = int(model.predict(X_model)[0])

    if pred == 1:
        st.error("🚨 ANOMALY DETECTED (label = 1)")
    else:
        st.success("✅ NORMAL PROCESS (label = 0)")

except Exception as e:
    st.exception(e)
