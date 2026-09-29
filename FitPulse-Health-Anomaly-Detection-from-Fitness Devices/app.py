import streamlit as st
import pandas as pd
import requests
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FitPulse Health Analytics Platform", layout="wide")

# ================= AUTO START BACKEND =================
import subprocess
import socket
import time
import sys

# Propagate secrets to os.environ so backend inherits it
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

if not os.getenv("OPENAI_API_KEY"):
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
        except Exception:
            pass

def _is_backend_alive(host="127.0.0.1", port=8003):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

@st.cache_resource
def ensure_backend_started():
    if not _is_backend_alive():
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "-m", "uvicorn", "backend:app", "--host", "127.0.0.1", "--port", "8003"]
        try:
            subprocess.Popen(
                cmd,
                cwd=backend_dir,
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            for _ in range(30):
                if _is_backend_alive():
                    break
                time.sleep(0.5)
        except Exception as e:
            print("Auto-start backend exception:", e)
    return True

ensure_backend_started()

# ================= CONFIG =================
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8003")
HEADERS = {"ngrok-skip-browser-warning": "true"}

# ================= CUSTOM UI STYLE =================
st.markdown("""
<style>

/* ----------- APP BACKGROUND ----------- */
.stApp {
    background-color: #F4FAF9;
}

/* ----------- HEADINGS ----------- */
h1, h2, h3 {
    color: #0F2F36;
    font-weight: 700;
}

/* ----------- METRIC CARDS ----------- */
.metric-card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #D7ECEB;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    text-align: center;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}

/* ----------- SECTION BOX ----------- */
.section-box {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #D7ECEB;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* ----------- SIDEBAR ----------- */
section[data-testid="stSidebar"] {
    background-color: #E0F2F1;
}

/* ----------- BUTTONS ----------- */
.stButton > button {
    background-color: #1B9AAA;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #157F87;
}

/* ----------- ALERT / ISSUE CARDS ----------- */
.issue-card {
    background: #FFF4F4;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #E63946;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ----------- TABLE ----------- */
[data-testid="stData hookup Frame"] {
    border-radius: 12px;
    border: 1px solid #D7ECEB;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Hide radio circles */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Radio item container */
    div[role="radiogroup"] label {
        padding: 6px 10px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 15px;
        cursor: pointer;
    }

    /* Hover */
    div[role="radiogroup"] label:hover {
        background-color: #f2f2f2;
    }

    /* Active item */
    div[role="radiogroup"] input:checked + div {
        background-color: #e6e6e6;
        font-weight: 600;
        border-radius: 6px;
        padding: 6px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ================= SESSION STATE =================
if "preprocess_done" not in st.session_state:
    st.session_state.preprocess_done = False

if "module2_done" not in st.session_state:
    st.session_state.module2_done = False

if "page" not in st.session_state:
    st.session_state.page = "Welcome"

if "redirect_page" not in st.session_state:
    st.session_state.redirect_page = None


# ================= API HELPERS =================
def api_get(endpoint, raw=False):
    try:
        r = requests.get(f"{BACKEND}{endpoint}", headers=HEADERS)
        if raw:
            return r
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint, files=None):
    try:
        return requests.post(f"{BACKEND}{endpoint}", files=files, headers=HEADERS).json()
    except Exception as e:
        return {"error": str(e)}


def api_post_json(endpoint, payload=None):
    try:
        return requests.post(f"{BACKEND}{endpoint}", json=payload, headers=HEADERS).json()
    except Exception as e:
        return {"error": str(e)}

# ================= SIDEBAR =================
st.sidebar.markdown("## FitPulse Controls")
st.sidebar.divider()

pages = [
    "Welcome",
    "Data Upload & Preprocessing",
    "Feature Extraction",
    "Trends",
    "Anomalies",
    "Distributions & DBSCAN",
    "User Dashboard",
    "Wellness Calculator",
    "AI Health Assistant",
    "Downloads"
]

if "page" not in st.session_state:
    st.session_state.page = "User Dashboard"

page = st.sidebar.radio(
    label="",
    options=pages,
    index=pages.index(st.session_state.page),
    label_visibility="collapsed"
)

st.session_state.page = page


# ================= LOAD DATA =================
# ================= LOAD DATA =================
# @st.cache_data
# def load_processed_data():
#     res = api_get("/dataframe")
#     if not res or "rows" not in res:
#         return pd.DataFrame()
#     return pd.DataFrame(res["rows"])

# df = load_processed_data()

# if not df.empty:
#     df["user_id"] = df["user_id"].astype(str)
#     df["date"] = pd.to_datetime(df["date"])


# ================= PAGE : WELCOME =================
if page == "Welcome":

    # ================= HERO =================
    st.markdown("""
    <style>
    .hero-wrap {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 30px;
    }

    .hero-wrap img {
        width: 100%;
        height: 420px;
        object-fit: cover;
        filter: brightness(45%);
    }

    .hero-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        width: 85%;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 800;
    }

    .hero-sub {
        margin-top: 12px;
        font-size: 18px;
    }
    </style>

    <div class="hero-wrap">
        <img src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d">
        <div class="hero-content">
            <div class="hero-title">Smarter Wearable Health Intelligence</div>
            <div class="hero-sub">
                Detect risks earlier. Understand behavior deeper.
                Deliver personalized AI-driven care.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= KPIs =================
    st.markdown("""
    <style>
    .kpi-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        border: 1px solid #E6F2F2;
        transition: 0.25s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 22px rgba(0,0,0,0.08);
    }

    .kpi-title {
        font-size: 15px;
        color: #457B9D;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F2F36;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Monitoring</div>
            <div class="kpi-value">24×7</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">AI Models</div>
            <div class="kpi-value">Running</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Users</div>
            <div class="kpi-value">Multi</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Reports</div>
            <div class="kpi-value">Instant</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ================= BUTTONS =================
    st.subheader("Quick Actions")

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("📤 Upload Data"):
            st.session_state.redirect_page = "Data Upload & Preprocessing"
            st.rerun()

    with b2:
        if st.button("📊 Open Dashboard"):
            st.session_state.redirect_page = "User Dashboard"
            st.rerun()

    with b3:
        if st.button("🤖 Ask AI"):
            st.session_state.redirect_page = "AI Health Assistant"
            st.rerun()



# ================= PAGE 1 =================
if page == "Data Upload & Preprocessing":

    st.title("FitPulse Health Analytics")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.markdown("### 📥 Need sample data to test?")
    st.caption("Download the pre-formatted health dataset below, or click **Use Built-in Sample Data** to start immediately:")
    
    dcol1, dcol2, dcol3 = st.columns([1, 1, 1.3])
    sample_csv_path = os.path.join(os.path.dirname(__file__), "sample_health_data.csv")
    sample_json_path = os.path.join(os.path.dirname(__file__), "sample_health_data.json")

    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            dcol1.download_button("📥 Download CSV", f, file_name="sample_health_data.csv", mime="text/csv", use_container_width=True)

    if os.path.exists(sample_json_path):
        with open(sample_json_path, "rb") as f:
            dcol2.download_button("📥 Download JSON", f, file_name="sample_health_data.json", mime="application/json", use_container_width=True)

    quick_load = dcol3.button("⚡ Use Built-in Sample Data", use_container_width=True)

    st.divider()
    uploaded = st.file_uploader("Or Upload Your Own CSV or JSON", type=["csv", "json"])

    if quick_load and os.path.exists(sample_csv_path):
        with st.spinner("Processing built-in sample data..."):
            with open(sample_csv_path, "rb") as f:
                res = api_post("/preprocess", files={"file": ("sample_health_data.csv", f, "text/csv")})
            if res.get("status") == "success":
                st.session_state.preprocess_done = True
                st.session_state.module2_done = False
                st.success("✅ Sample dataset preprocessed successfully!")
                ov = res["overview"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Records", ov["rows_loaded"])
                c2.metric("Users", ov["users"])
                c3.metric("Days", ov["days"])
                c4.metric("Average Heart Rate", ov["avg_hr"])
                st.dataframe(pd.DataFrame(res["preview"]), use_container_width=True)
            else:
                st.error(f"Error processing sample data: {res.get('error')}")

    if st.button("Run Preprocessing"):

        if uploaded:
            with st.spinner("Processing data..."):
                res = api_post("/preprocess", files={"file": uploaded})

            if res.get("status") == "success":
                st.session_state.preprocess_done = True
                st.session_state.module2_done = False   # ⭐ RESET FEATURES FOR NEW DATA

                ov = res["overview"]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Records", ov["rows_loaded"])
                c2.metric("Users", ov["users"])
                c3.metric("Days", ov["days"])
                c4.metric("Average Heart Rate", ov["avg_hr"])

                st.dataframe(pd.DataFrame(res["preview"]), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 2 =================
elif page == "Feature Extraction":

    st.title("Feature Extraction")

    if not st.session_state.preprocess_done:
        st.warning("Run preprocessing first")

    else:
        # ✅ ALWAYS RECOMPUTE
        with st.spinner("Extracting features..."):
            res = api_post_json("/module2")

        if res.get("status") == "success":
            st.session_state.module2_done = True
            st.success("Feature extraction completed")
        else:
            st.error(res.get("error", "Module2 failed"))

        # ✅ LOAD TSFRESH
        ts = api_get("/module3/tsfresh-summary")
        df_ts = pd.DataFrame(ts.get("features", []))

        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        if df_ts.empty:
            st.info("No TSFRESH features available")
        else:
            fig = px.bar(
                df_ts,
                x="description",
                y="importance",
                color="description",
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            # ✅ INCREASE BAR WIDTH HERE
            fig.update_traces(width=0.8)

            fig.update_layout(
                showlegend=False,
                xaxis_title="Feature",
                yaxis_title="Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_ts, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 3 =================
elif page == "Trends":

    if not st.session_state.module2_done:
        st.warning("Run feature extraction first")
    else:
        st.title("Health Trends")

        overview = api_get("/overview")
        users = ["All"] + overview.get("users_list", [])
        user = st.selectbox("Select User", users)

        metrics = st.multiselect(
            "Select Metrics",
            ["heart_rate", "steps", "sleep"],
            default=["heart_rate", "steps", "sleep"]
        )

        for metric in metrics:
            r = api_get(f"/module3/prophet/{metric}?user_id={user}", raw=True)

            if r.status_code == 200:
                st.image(io.BytesIO(r.content), use_container_width=True)

# ================= PAGE 4 =================
elif page == "Anomalies":

    st.title("Anomaly Analysis")

    overview = api_get("/overview")
    users = ["All"] + overview.get("users_list", [])
    user = st.selectbox("Select User", users)

    summary = api_get(f"/module3/summary?user_id={user}")

    if summary.get("summary"):
        df_summary = pd.DataFrame(
            summary["summary"].items(),
            columns=["Metric", "Count"]
        )

        # ✅ STEP 1: Create Severity column
        def classify_severity(count):
            if count >= 10:
                return "High"
            elif count >= 5:
                return "Medium"
            else:
                return "Low"

        df_summary["Severity"] = df_summary["Count"].apply(classify_severity)
    fig = px.bar(
    df_summary,
    x="Metric",
    y="Count",
    color="Metric",
    color_discrete_map={
        "sleep_high": "#9B5DE5",
        "sleep_low": "#00BBF9",
        "dbscan_outlier": "#EF476F",
        "heart_rate_high": "#F77F00",
        "heart_rate_low": "#577590",
        "low_activity": "#43AA8B"
        }
    )

    # 🔥 MAKE BARS THICK (same as Feature Extraction)
    fig.update_traces(
        width=0.8,
        texttemplate="%{y}",
        textposition="outside"
    )

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        xaxis_title="Health Metric",
        yaxis_title="Anomaly Count",
        legend_title="Anomaly Type"
    )

    st.plotly_chart(fig, use_container_width=True)


    table = api_get(f"/module3/anomaly-with-recommendations?user_id={user}")
    if not table:
      st.error("Backend not responding.")
    elif not table.get("rows"):
      st.info("No anomalies detected.")
    else:
      st.dataframe(pd.DataFrame(table["rows"]), use_container_width=True)


# ================= PAGE 5 =================
elif page == "Distributions & DBSCAN":

    if not st.session_state.module2_done:
        st.warning("Run feature extraction first")
    else:
        st.title("Distributions & Clustering")

        overview = api_get("/overview")
        users = ["All"] + overview.get("users_list", [])
        user = st.selectbox("Select User", users)

        metrics = ["heart_rate", "steps", "sleep"]

        # ----- Distribution Charts -----
        st.subheader("Metric Distributions")

        for metric in metrics:
            r = api_get(f"/module3/distribution/{metric}?user_id={user}", raw=True)

            if r.status_code == 200:
                st.image(io.BytesIO(r.content), use_container_width=True)

        st.divider()

        # ----- DBSCAN -----
        st.subheader("DBSCAN Clustering")

        r = api_get(f"/module3/dbscan?user_id={user}", raw=True)

        if r.status_code == 200:
            st.image(io.BytesIO(r.content), use_container_width=True)

# ================= PAGE 6 : USER DASHBOARD =================
elif page == "User Dashboard":

    st.title("User Health Dashboard")

    # ---------- CARD CSS ----------
    st.markdown("""
    <style>
    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        text-align: center;
        transition: 0.3s;
    }

    .metric-card:hover {
        transform: translateY(-4px);
    }

    .metric-title {
        font-size: 16px;
        color: #457B9D;
        font-weight: 600;
    }

    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1D3557;
        margin-top: 8px;
    }

    .issue-card {
        background: white;
        padding: 16px;
        border-radius: 14px;
        border-left: 6px solid #E63946;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------- LOAD DATA ----------
    res = api_get("/dataframe")

    if "rows" not in res or not res["rows"]:
        st.warning("Run preprocessing first.")
        st.stop()

    df = pd.DataFrame(res["rows"])
    df["date"] = pd.to_datetime(df["date"])


    # ---------- COLUMN ALIGNMENT ----------
    if "avg_heart_rate" in df.columns:
        df.rename(columns={"avg_heart_rate": "heart_rate"}, inplace=True)

    if "TotalSteps" in df.columns:
        df.rename(columns={"TotalSteps": "steps"}, inplace=True)

    if "total_sleep_minutes" in df.columns:
        df["sleep"] = df["total_sleep_minutes"] / 60

    # ---------- USER SELECT ----------
    user_ids = df["user_id"].astype(str).unique()
    selected_user = st.selectbox("Select User", user_ids)
    st.session_state.selected_user = selected_user

    user_data = df[df["user_id"].astype(str) == selected_user].copy()

    if user_data.empty:
        st.warning("No data available for this user.")
        st.stop()

    # ---------- HEALTH SCORE ----------
    health = api_get(f"/module3/health-score?user_id={selected_user}") or {}

    components = health.get("components", {})
    hr_score = components.get("heart_rate", 0)
    steps_score = components.get("steps", 0)
    sleep_score = components.get("sleep", 0)
    health_score = health.get("health_score", 0)

    # ---------- METRIC CARDS ----------
    avg_heart = round(user_data["heart_rate"].mean(), 1)
    avg_steps = round(user_data["steps"].mean(), 0)
    avg_sleep = round(user_data["sleep"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-title"> Avg Heart Rate</div>
        <div class="metric-value">{avg_heart} BPM</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-title"> Avg Steps</div>
        <div class="metric-value">{avg_steps}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-title"> Avg Sleep</div>
        <div class="metric-value">{avg_sleep} hrs</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-title"> Health Score</div>
        <div class="metric-value">{health_score}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------- DONUT CHART ----------
    st.subheader("Health Score Breakdown")

    fig = go.Figure(data=[go.Pie(
        labels=["Heart Rate", "Steps", "Sleep"],
        values=[hr_score, steps_score, sleep_score],
        hole=0.6
    )])

    fig.update_layout(
        annotations=[dict(text=f"{health_score}%", x=0.5, y=0.5, showarrow=False)]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------- ANOMALIES ----------
    st.subheader("Health Issues & Recommendations")

    anomalies = api_get(
        f"/module3/anomaly-with-recommendations?user_id={selected_user}"
    )

    if anomalies and anomalies.get("rows"):

        for row in anomalies["rows"]:
            st.markdown(f"""
            <div class="issue-card">
                <b>{row.get("issue")}</b><br>
                Severity: {row.get("severity")} <br><br>
                💡 {row.get("recommendation")}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.success("No major health risks detected.")

    st.divider()

    # ---------- TRENDS ----------
    st.subheader("Health Trends")

    metrics = ["heart_rate", "steps", "sleep"]

    for metric in metrics:

        if metric in user_data.columns:

            user_data[f"{metric}_rolling"] = (
                user_data[metric].rolling(3, min_periods=1).mean()
            )

            fig = px.line(
                user_data,
                x="date",
                y=[metric, f"{metric}_rolling"],
                title=f"{metric.replace('_',' ').title()} Trend"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------- DBSCAN CLUSTER ----------
    st.subheader("Activity Clustering")

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN

        cluster_cols = ["heart_rate", "steps", "sleep"]

        if all(col in user_data.columns for col in cluster_cols):

            X = StandardScaler().fit_transform(user_data[cluster_cols])
            db = DBSCAN(eps=1.5, min_samples=2).fit(X)

            user_data["cluster"] = db.labels_

            fig = px.scatter_3d(
                user_data,
                x="heart_rate",
                y="steps",
                z="sleep",
                color="cluster",
                title="User Activity Clusters"
            )

            st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("Not enough data for clustering.")
# ================= PAGE 7 =================
# ================= PAGE 7 =================
elif page == "Downloads":

    st.title("Download Reports")

    st.markdown("""
    Welcome to the **Report Download Center**

    Here you can generate and download detailed health analytics reports based on
    processed wearable data. These reports include insights, anomaly detection,
    clustering analysis, and personalized health recommendations.
    """)

    st.divider()

    # ---------- FULL DASHBOARD PDF ----------
    st.subheader(" Full Dashboard Report")

    st.markdown("""
    This report contains:
    - Dataset overview and statistics
    - Behavioral pattern insights
    - Anomaly detection summary
    - Visual trend analytics
    - Health recommendations
    """)

    if st.button("Generate & Download Full Dashboard PDF"):

        r = requests.get(f"{BACKEND}/download-report", headers=HEADERS)

        if r.status_code == 200:
            st.download_button(
                "⬇ Download Full Report",
                r.content,
                file_name="fitpulse_dashboard_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to generate report")

    st.divider()

    # ---------- USER REPORT ----------
    st.subheader(" Personalized User Report")

    st.markdown("""
    This report provides **user-specific health insights**, including:

    - Individual health score breakdown
    - Personalized anomaly detection
    - Lifestyle recommendations
    - Activity trend visualization
    - Recent activity summary
    """)

    selected_user = st.session_state.get("selected_user")

    if not selected_user:
        st.warning("Please select a user from the Dashboard page first.")
    else:
        if st.button("Generate User Report"):

            r = requests.get(
                f"{BACKEND}/module4/user-dashboard-report?user_id={selected_user}",
                headers=HEADERS
            )

            if r.status_code == 200:
                st.download_button(
                    "⬇ Download User Report",
                    r.content,
                    file_name=f"user_{selected_user}_dashboard.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate user report")

# ================= AI CHATBOT =================
elif page == "AI Health Assistant":

    st.title("AI Health Assistant")

    st.markdown("Ask questions about heart rate, sleep, activity, risks, or improvement tips.")

    # Conversation memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    prompt = st.chat_input("Type your health question...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # send selected user for personalization
        user_id = st.session_state.get("selected_user", "All")

        with st.spinner("Thinking..."):
            res = api_post_json("/module5/chat", {
                "question": prompt,
                "user_id": user_id
            })

        answer = res.get("answer", "Sorry, I could not respond.")

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})



# ================= PAGE : WELLNESS CALCULATOR =================
elif page == "Wellness Calculator":

    st.title("Wellness Assessment & Daily Target Estimator")

    st.markdown("""
    This tool evaluates overall wellness using cardiovascular,
    activity, and recovery indicators.
    Values can be entered manually or derived from historical records.
    """)

    # ======================================================
    # MODE SELECTION
    # ======================================================
    mode = st.radio(
        "Select Data Source",
        ["Use Manual Input", "Use Dataset Averages"]
    )

    # ======================================================
    # MANUAL INPUT MODE
    # ======================================================
    if mode == "Use Manual Input":

        col1, col2 = st.columns(2)

        with col1:
            resting_hr = st.number_input("Resting Heart Rate (BPM)", 30, 150, 70)

        with col2:
            daily_steps = st.number_input("Average Daily Steps", 0, 50000, 6000)
            sleep_hrs = st.number_input(
                "Average Sleep Duration (hours)", 0.0, 24.0, 7.0, step=0.5
            )

        st.divider()

        if st.button("Run Assessment"):

            hr_score = max(0, 100 - abs(resting_hr - 70))
            steps_score = min(100, (daily_steps / 10000) * 100)
            sleep_score = max(0, 100 - abs(sleep_hrs - 8) * 12.5)

            overall_score = round((hr_score + steps_score + sleep_score) / 3, 1)

            if overall_score >= 80:
                status = "Within healthy range"
            elif overall_score >= 50:
                status = "Moderate deviation"
            else:
                status = "High deviation"

            st.subheader("Assessment Outcome")
            st.write(f"Composite Wellness Score: **{overall_score}%**")
            st.write(f"Interpretation: **{status}**")

            st.divider()

            st.subheader("Component Contribution")

            fig = go.Figure(data=[go.Pie(
                labels=["Heart Rate", "Steps", "Sleep"],
                values=[hr_score, steps_score, sleep_score],
                hole=0.5
            )])

            fig.update_layout(
                annotations=[dict(text=f"{overall_score}%", x=0.5, y=0.5, showarrow=False)]
            )

            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # DATASET MODE  → CALL BACKEND
    # ======================================================
    else:

        res = api_get("/dataframe")

        if "rows" not in res or not res["rows"]:
            st.warning("Dataset not available. Run preprocessing first.")
            st.stop()

        df = pd.DataFrame(res["rows"])
        df["user_id"] = df["user_id"].astype(str)

        users = sorted(df["user_id"].unique())
        selected_user = st.selectbox("Select User", users)

        st.info("Scores are computed from backend using historical averages.")

        st.divider()

        if st.button("Run Assessment"):

            health = api_get(f"/module3/health-score?user_id={selected_user}")

            if not health or "health_score" not in health:
                st.error("Unable to calculate score.")
                st.stop()

            overall_score = health.get("health_score", 0)
            status = health.get("status", "")

            components = health.get("components", {})
            hr_score = components.get("heart_rate", 0)
            steps_score = components.get("steps", 0)
            sleep_score = components.get("sleep", 0)

            st.subheader("Assessment Outcome")
            st.write(f"Composite Wellness Score: **{overall_score}%**")
            st.write(f"Interpretation: **{status}**")

            st.divider()

            st.subheader("Component Contribution")

            fig = go.Figure(data=[go.Pie(
                labels=["Heart Rate", "Steps", "Sleep"],
                values=[hr_score, steps_score, sleep_score],
                hole=0.5
            )])

            fig.update_layout(
                annotations=[dict(text=f"{overall_score}%", x=0.5, y=0.5, showarrow=False)]
            )

            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # NEW SECTION : PERSONAL TARGETS
    # ======================================================
    st.divider()
    st.header("Personal Daily Targets")

    st.markdown("Set lifestyle inputs to estimate step goals and ideal sleep timing.")

    # ======================================================
    # STEP GOAL CALCULATOR
    # ======================================================
    st.subheader("Daily Step Goal Estimator")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 5, 100, 30)

    with col2:
        weight = st.number_input("Weight (kg)", 20, 200, 65)

    with col3:
        activity = st.selectbox(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Active", "Very Active"]
        )

    if st.button("Calculate Step Goal"):

        goal = 6000

        if age < 30:
            goal += 2000
        elif age < 50:
            goal += 1000
        else:
            goal -= 500

        if weight > 80:
            goal += 1000

        multiplier = {
            "Sedentary": 0.9,
            "Lightly Active": 1.0,
            "Active": 1.2,
            "Very Active": 1.4
        }

        final_goal = int(goal * multiplier[activity])

        st.success(f"Recommended Daily Steps: **{final_goal} steps/day**")

    st.divider()

    # ======================================================
    # SLEEP CYCLE PLANNER
    # ======================================================
    st.subheader("Sleep Cycle Planner")

    wake_time = st.time_input("Wake-up time")

    if st.button("Suggest Bedtime"):

        from datetime import datetime, timedelta

        today = datetime.today()
        wake_dt = datetime.combine(today.date(), wake_time)

        bed_5 = wake_dt - timedelta(minutes=90 * 5)
        bed_6 = wake_dt - timedelta(minutes=90 * 6)

        st.success("Recommended times to fall asleep:")
        st.write(f"• 5 cycles: **{bed_5.strftime('%I:%M %p')}**")
        st.write(f"• 6 cycles: **{bed_6.strftime('%I:%M %p')}**")


        st.divider()
    st.header("Advanced Body & Recovery Intelligence")

    # ======================================================
    # BMI + BODY STATUS
    # ======================================================
    st.subheader("BMI & Weight Category")

    hcol1, hcol2 = st.columns(2)

    with hcol1:
        height_cm = st.number_input("Height (cm)", 100, 230, 165)

    with hcol2:
        weight_kg = st.number_input("Current Weight (kg)", 30, 200, 65, key="bmi_weight")

    if st.button("Calculate BMI"):

        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m ** 2), 1)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        st.success(f"BMI: **{bmi}** → {category}")

    st.divider()

    # ======================================================
    # CALORIE BURN ESTIMATOR (VERY POPULAR)
    # ======================================================
    st.subheader("Daily Calorie Burn Estimate")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age_cal = st.number_input("Age (years)", 10, 100, 30, key="cal_age")

    if st.button("Estimate Calories"):

        # Mifflin-St Jeor formula
        if gender == "Male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_cal + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_cal - 161

        calories = int(bmr * 1.3)  # light daily movement assumption

        st.success(f"Estimated daily burn: **{calories} kcal/day**")

    st.divider()

    # ======================================================
    # WEIGHT TARGET PLANNER
    # ======================================================


    st.subheader("AI Weight Goal Planner")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100, 30, key="ai_age")
        height = st.number_input("Height (cm)", 120, 220, 165, key="ai_height")

    with col2:
        weight = st.number_input("Weight (kg)", 30, 200, 70, key="ai_weight")
        weeks = st.slider("Timeline (weeks)", 2, 24, 8, key="ai_weeks")

    activity = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Active", "Very Active"],
        key="ai_activity"
    )

    goal = st.selectbox(
        "Goal",
        ["Lose weight", "Gain weight", "Maintain weight"],
        key="ai_goal"
    )

    if st.button("Generate AI Plan", key="ai_plan_btn"):

        with st.spinner("AI coach is preparing your plan..."):

            res = api_post_json(
                "/ai/weight-plan",
                {
                    "age": age,
                    "height": height,
                    "weight": weight,
                    "activity": activity,
                    "goal": goal,
                    "weeks": weeks
                }
            )

        st.success("Your Personalized Plan")
        st.write(res.get("plan", "No plan generated"))







st.caption("FitPulse | FastAPI Backend with Streamlit Frontend")