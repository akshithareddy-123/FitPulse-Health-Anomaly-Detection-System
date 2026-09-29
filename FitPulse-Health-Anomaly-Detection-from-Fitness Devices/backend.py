

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import os
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    HAS_TSFRESH = True
except Exception:
    extract_features = None
    impute = None
    HAS_TSFRESH = False
from fastapi import HTTPException
import traceback

from fastapi import Body
from typing import Optional

# import google.generativeai as genai

# genai.configure(api_key="AIzaSyDL-SNYgzASQV_OXxw6N-fOXf6HhZ8VgWA")
# model = genai.GenerativeModel("gemini-1.5-flash")

from openai import OpenAI
import os

import requests

def _load_env():
    for d in [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        for fname in [".env", os.path.join(".streamlit", "secrets.toml")]:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip("'\"")
                                if k in ["OPENAI_API_KEY", "GEMINI_API_KEY"] and v:
                                    os.environ[k] = v
                except Exception:
                    pass

_load_env()

def call_gemini_api(prompt):
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        _load_env()
        key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None

    # Try SDK first
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        for mname in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
            try:
                model = genai.GenerativeModel(mname)
                resp = model.generate_content(prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                continue
    except Exception as e:
        print("Gemini SDK error:", e)

    # Fallback to direct REST
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        r = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=12
        )
        if r.status_code == 200:
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print("Gemini REST error:", e)

    return None

def get_client():
    k = os.getenv("OPENAI_API_KEY", "")
    if not k or k == "my_auth_token":
        _load_env()
        k = os.getenv("OPENAI_API_KEY", "")
    return OpenAI(api_key=k if k and k != "my_auth_token" else "none")


# ================= APP =================
app = FastAPI(title="FitPulse Backend")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= GLOBAL STATE =================
CLEAN_DF = None
FEATURE_DF = None
ANOMALY_DF = None
RULE_RECOMMENDATIONS = None

# ================= ROOT =================
def classify_severity(count: int):
    if count >= 20:
        return "High"
    elif count >= 5:
        return "Medium"
    return "Low"

def df_to_table(df, max_rows=10):
    data = [df.columns.tolist()] + df.head(max_rows).values.tolist()
    return Table(data, repeatRows=1)

# ======================================================
# AUTO COLUMN MAPPER
# ======================================================
def normalize_columns(df):
    mapping = {}

    for col in df.columns:
        c = col.lower().strip().replace(" ", "").replace("_", "")

        # user id
        if c in ["userid", "user", "id", "memberid"]:
            mapping[col] = "user_id"

        # date
        elif c in ["date", "day", "datetime", "timestamp"]:
            mapping[col] = "date"

        # heart rate
        elif c in ["heartrate", "avgheartrate", "pulse", "hr", "bpm"]:
            mapping[col] = "avg_heart_rate"

        # steps
        elif c in ["steps", "totalsteps", "stepcount", "stepscount"]:
            mapping[col] = "TotalSteps"

        # sleep
        elif c in [
            "sleep", "sleepminutes", "totalsleepminutes",
            "minutesasleep", "sleepduration"
        ]:
            mapping[col] = "total_sleep_minutes"

    df = df.rename(columns=mapping)
    return df

@app.get("/")
def root():
    return {"message": "FitPulse Backend is running"}

# ================= PREPROCESSING (CSV + JSON) =================
@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    global CLEAN_DF

    try:
        if file.filename.endswith(".csv"):
          df = pd.read_csv(file.file)
        elif file.filename.endswith(".json"):
         df = pd.read_json(file.file)
        else:
         return JSONResponse(status_code=400, content={"error": "Only CSV or JSON supported"})

        # ✅ AUTO FIX COLUMN NAMES
        df = normalize_columns(df)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid file: {e}"})

    required_cols = ["user_id", "date", "TotalSteps", "avg_heart_rate", "total_sleep_minutes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return JSONResponse(status_code=400, content={"error": f"Missing columns: {missing}"})

    # ---------- CLEANING ----------
    # ---------- CLEANING ----------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["TotalSteps", "avg_heart_rate", "total_sleep_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSteps"].fillna(0, inplace=True)
    df["avg_heart_rate"].fillna(df["avg_heart_rate"].median(), inplace=True)

    # ✅ SAFE sleep handling
    df["total_sleep_minutes"].fillna(0, inplace=True)

    # ---------- AGGREGATE (FIXED) ----------
    df = df.groupby(["user_id", "date"], as_index=False).agg({
        "TotalSteps": "sum",
        "avg_heart_rate": "mean",
        "total_sleep_minutes": "sum"   # ✅ SUM, NOT MEAN
    })


    # ---------- RENAME FOR UI ----------
    df.rename(columns={
        "avg_heart_rate": "heart_rate",
        "TotalSteps": "steps",
        "total_sleep_minutes": "sleep"
    }, inplace=True)

    # Convert sleep minutes → hours (matches screenshot ~7.05)
    df["sleep"] = (df["sleep"] / 60).round(2)

    CLEAN_DF = df
    df.to_csv("clean_data.csv", index=False)


    return {
        "status": "success",
        "overview": {
            "rows_loaded": len(df),
            "users": df["user_id"].nunique(),
            "days": df["date"].nunique(),
            "avg_hr": round(df["heart_rate"].mean(), 1)
        },
        "preview": df.to_dict(orient="records")
    }



# ================= OVERVIEW =================
@app.get("/overview")
def overview():
    if CLEAN_DF is None:
        return {"error": "Run /preprocess first"}
    df = CLEAN_DF
    return {
        "rows": len(df),
        "users_count": df["user_id"].nunique(),
        "users_list": df["user_id"].astype(str).unique().tolist(),  # <-- add this
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "avg_heart_rate": round(df["heart_rate"].mean(), 2),
        "avg_steps": round(df["steps"].mean(), 2),
        "avg_sleep_hours": round(df["sleep"].mean(), 2),
    }
@app.get("/dataframe")
def get_clean_dataframe():
    if CLEAN_DF is None or CLEAN_DF.empty:
        return {"rows": []}

    df = CLEAN_DF.copy()
    df["date"] = df["date"].astype(str)
    df["user_id"] = df["user_id"].astype(str)

    return {"rows": df.to_dict(orient="records")}

# ================= FEATURES + ANOMALIES =================
@app.post("/module2")
def module2():
    global FEATURE_DF, ANOMALY_DF, RULE_RECOMMENDATIONS

    if CLEAN_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /preprocess first"})

    df = CLEAN_DF.copy().sort_values(["user_id", "date"])

    # =====================================================
    # TSFRESH FEATURES (Single User Sliding Window Method)
    # =====================================================
    try:
        window_size = 7

        # ---------- Create Sliding Windows ----------
        windows = []
        if HAS_TSFRESH and len(df) >= window_size:

            for i in range(len(df) - window_size + 1):
                win = df.iloc[i:i + window_size].copy()
                win["window_id"] = i
                windows.append(win)

            win_df = pd.concat(windows)

            ts_df = win_df.melt(
                id_vars=["window_id", "date"],
                value_vars=["heart_rate", "steps", "sleep"],
                var_name="metric",
                value_name="value"
            )

            ts_features = extract_features(
                ts_df,
                column_id="window_id",
                column_sort="date",
                column_kind="metric",
                column_value="value",
                disable_progressbar=True
            )

            impute(ts_features)

            TSFRESH_DF = ts_features.reset_index()

        else:
            # ---------- Fallback for very small dataset ----------
            TSFRESH_DF = df.groupby("user_id").agg({
                "heart_rate": ["mean", "std", "min", "max"],
                "steps": ["mean", "std", "min", "max"],
                "sleep": ["mean", "std", "min", "max"]
            })

            TSFRESH_DF.columns = ["_".join(c) for c in TSFRESH_DF.columns]
            TSFRESH_DF.reset_index(inplace=True)

        TSFRESH_DF.to_csv("tsfresh_features.csv", index=False)

    except Exception as e:
        print("TSFRESH failed:", str(e))
        traceback.print_exc()

    # =====================================================
    # ROLLING FEATURES
    # =====================================================
    FEATURE_DF = df.copy()

    for col in ["steps", "heart_rate", "sleep"]:
        FEATURE_DF[f"{col}_mean_7"] = FEATURE_DF.groupby("user_id")[col].rolling(7).mean().reset_index(0, drop=True)
        FEATURE_DF[f"{col}_std_7"] = FEATURE_DF.groupby("user_id")[col].rolling(7).std().reset_index(0, drop=True)

    FEATURE_DF.to_csv("feature_data.csv", index=False)

    # Rule-based anomalies
    records = []
    for _, r in df.iterrows():
        if r["heart_rate"] > 120:
            records.append((r["user_id"], r["date"], "heart_rate_high"))

        if r["heart_rate"] < 40:
            records.append((r["user_id"], r["date"], "heart_rate_low"))

        if r["steps"] <2000:
            records.append((r["user_id"], r["date"], "low_activity"))

        if 0 < r["sleep"] < 3:
            records.append((r["user_id"], r["date"], "sleep_low"))

        if r["sleep"] > 10:
            records.append((r["user_id"], r["date"], "sleep_high"))

    rule_df = pd.DataFrame(records, columns=["user_id", "date", "metric"])

    # DBSCAN anomalies
    X_scaled = StandardScaler().fit_transform(df[["heart_rate", "steps", "sleep"]])
    labels = DBSCAN(eps=0.8, min_samples=2).fit_predict(X_scaled)
    df["cluster"] = labels
    dbscan_df = df[df["cluster"] == -1][["user_id", "date"]].copy()
    dbscan_df["metric"] = "dbscan_outlier"

    ANOMALY_DF = pd.concat([rule_df, dbscan_df], ignore_index=True)

    summary_df = (
        ANOMALY_DF
        .groupby(["user_id", "metric"])
        .size()
        .reset_index(name="count")
    )

    summary_df["severity"] = summary_df["count"].apply(classify_severity)

    # Save both versions
    ANOMALY_DF.to_csv("anomaly_raw.csv", index=False)
    summary_df.to_csv("anomaly_report.csv", index=False)


    recs = []


    for _, row in summary_df.iterrows():
        user = row["user_id"]
        metric = row["metric"]
        severity = row["severity"]

        if metric == "heart_rate_high":
            recs.append({
                "user_id": user,
                "issue": "High heart rate",
                "severity": severity,
                "recommendation": "Reduce high-intensity workouts and consult a physician if persistent."
            })

        elif metric == "heart_rate_low":
            recs.append({
                "user_id": user,
                "issue": "Low heart rate",
                "severity": severity,
                "recommendation": "Ensure adequate nutrition and consult a healthcare professional."
            })

        elif metric == "no_steps":
            recs.append({
                "user_id": user,
                "issue": "No physical activity",
                "severity": severity,
                "recommendation": "Increase daily movement with light walks or stretching."
            })

        elif metric == "sleep_low":
            recs.append({
                "user_id": user,
                "issue": "Low sleep duration",
                "severity": severity,
                "recommendation": "Aim for at least 7–8 hours of consistent sleep."
            })

        elif metric == "sleep_high":
            recs.append({
                "user_id": user,
                "issue": "Excessive sleep duration",
                "severity": severity,
                "recommendation": "Excess sleep may indicate fatigue or health issues."
            })

        elif metric == "dbscan_outlier":
            recs.append({
                "user_id": user,
                "issue": "Unusual health pattern",
                "severity": severity,
                "recommendation": "Monitor trends closely; consider lifestyle adjustments."
            })

    RULE_RECOMMENDATIONS = pd.DataFrame(recs)
    RULE_RECOMMENDATIONS.to_csv("recommendations.csv", index=False)

    return {
        "status": "success",
        "total_anomalies": len(ANOMALY_DF),
        "summary_rows": len(summary_df)
    }
@app.get("/module3/feature-table")
def feature_table():
    global FEATURE_DF

    if FEATURE_DF is None or FEATURE_DF.empty:
        return {"rows": []}

    df = FEATURE_DF.copy()

    # ✅ 1. Convert datetime → string
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    # ✅ 2. Replace NaN / inf with None (JSON-safe)
    df = df.replace([np.nan, np.inf, -np.inf], None)

    # ✅ 3. Convert numpy types → Python native
    df = df.astype(object)

    rows = df.head(200).to_dict(orient="records")

    return {"rows": rows}




# ================= ANOMALY SUMMARY =================

@app.get("/module3/summary")
def anomaly_summary(user_id: str = "All"):
    if ANOMALY_DF is None or ANOMALY_DF.empty:
        return {"summary": {}}

    df = ANOMALY_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == str(user_id)]

    summary = (
        df.groupby("metric")
          .size()
          .to_dict()
    )

    return {"summary": summary}

def plot_prophet(df, column, fname, ylabel, future_days=7):
    df2 = df[["date", column]].rename(columns={"date": "ds", column: "y"})

    if len(df2) < 10:
        return None

    # ---------- TRAIN MODEL ----------
    model = Prophet()
    model.fit(df2)

    # ---------- CREATE FUTURE DATES ----------
    future = model.make_future_dataframe(periods=future_days)

    # ---------- FORECAST ----------
    forecast = model.predict(future)

    # ---------- MERGE ACTUAL DATA ----------

    merged = forecast.merge(df2, on="ds", how="left")

        # ================================
    # Overlay detected anomalies
    # ================================
    global ANOMALY_DF

    anomaly_dates = []

    if ANOMALY_DF is not None and not ANOMALY_DF.empty:
        anomaly_dates = (
            ANOMALY_DF["date"]
            .astype(str)
            .tolist()
        )

    merged["is_anomaly"] = merged["ds"].astype(str).isin(anomaly_dates)


    # Residuals only where actual data exists
    merged["residual"] = merged["y"] - merged["yhat"]

    threshold = 2.5 * merged["residual"].std()
    outliers = merged[abs(merged["residual"]) > threshold]

    # ---------- PLOT ----------
    plt.figure(figsize=(12, 5))

    # Actual values
    plt.plot(merged["ds"], merged["y"], label=f"Actual {ylabel}", color="blue")

    # Prophet prediction (Past + Future)
    plt.plot(merged["ds"], merged["yhat"], label="Prophet Prediction", color="orange")

    # Confidence interval
    plt.fill_between(
        merged["ds"],
        merged["yhat_lower"],
        merged["yhat_upper"],
        alpha=0.3,
        label="Confidence Interval"
    )

    # Highlight anomalies (only past actual points)
    # plt.scatter(outliers["ds"], outliers["y"], color="red", label="Anomaly")
        # Plot detected anomalies from module2
    anom_points = merged[merged["is_anomaly"]]

    if not anom_points.empty:
        plt.scatter(
            anom_points["ds"],
            anom_points["y"],
            color="red",
            s=70,
            label="Detected Anomaly"
        )

    # Show vertical line separating past vs future
    last_date = df2["ds"].max()
    plt.axvline(last_date, linestyle="--", color="black", label="Future Start")

    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Forecast with Prophet")
    plt.legend()
    plt.grid(True)

    plt.savefig(fname)
    plt.close()

    return fname


@app.get("/module3/tsfresh-summary")
def tsfresh_summary():
    try:
        if not os.path.exists("tsfresh_features.csv"):
            return JSONResponse(content={"features": []})

        df = pd.read_csv("tsfresh_features.csv")

        # If user_id exists, drop it for full-dataset summary
        if "user_id" in df.columns:
            df = df.drop(columns=["user_id"])

        # Only numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return JSONResponse(content={"features": []})

        # Top 10 features by variance
        var_series = numeric_df.var().dropna()
        if var_series.empty:
            return JSONResponse(content={"features": []})

        top_var = var_series.sort_values(ascending=False).head(10)

        feature_list = []
        for feat, var in top_var.items():
            feature_list.append({
                "description": humanize_tsfresh_feature(feat),
                "importance": float(np.log10(var + 1)),
                "feature": feat
            })

        return {"features": feature_list}

    except Exception:
        print("TSFRESH SUMMARY ERROR\n", traceback.format_exc())
        return JSONResponse(content={"features": []})





def humanize_tsfresh_feature(name):
    if "__c3__" in name:
        return "Non-linear step behavior from previous days"
    if "time_reversal_asymmetry" in name:
        return "Irregular activity trend"
    if "abs_energy" in name:
        return "Overall activity energy"
    if "change_quantiles" in name:
        return "Sudden activity changes"
    if "linear_trend" in name:
        return "Long-term activity trend"
    if "lag_" in name:
        return "Previous day effect"
    return "General activity pattern"

@app.get("/module3/health-score")
def health_score(user_id: str = "All"):
    if CLEAN_DF is None or CLEAN_DF.empty:
        return {"error": "No data available. Run /preprocess first."}

    df = CLEAN_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == str(user_id)]

    if df.empty:
        return {"error": "No data for selected user."}

    # ---------- Calculate continuous scores ----------
    hr_score = max(0, 100 - (df['heart_rate'] - 70).abs().mean())
    steps_score = min(100, df['steps'].mean() / 10000 * 100)
    sleep_score = 100 - abs(df['sleep'].mean() - 7.5) / 7.5 * 100

    # ---------- Final health score ----------
    health_score = round((hr_score + steps_score + sleep_score) / 3, 1)

    # Optional: classify as High/Medium/Low for donut chart
    if health_score >= 80:
        status = "High"
    elif health_score >= 50:
        status = "Medium"
    else:
        status = "Low"

    # ---------- Return JSON for frontend ----------
    return {
        "user_id": user_id,
        "health_score": health_score,
        "status": status,
        "components": {
            "heart_rate": round(hr_score, 1),
            "steps": round(steps_score, 1),
            "sleep": round(sleep_score, 1)
        }
    }

@app.get("/module3/prophet/{metric}")
def prophet_metric(metric: str, user_id: str = "All"):
    mapping = {
        "heart_rate": "Heart Rate (BPM)",
        "steps": "Steps",
        "sleep": "Sleep (Hours)"
    }

    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /module2 first"})

    if metric not in mapping:
        return JSONResponse(status_code=400, content={"error": "Invalid metric"})

    # ✅ FORCE STRING TYPES (THIS IS THE KEY FIX)
    df = FEATURE_DF.copy()
    df["user_id"] = df["user_id"].astype(str)
    user_id = str(user_id)

    # ✅ APPLY USER FILTER
    if user_id != "All":
        df = df[df["user_id"] == user_id]

    if df.empty:
        return JSONResponse(status_code=400, content={"error": "No data for selected user"})

    fname = f"prophet_{metric}.png"

    # ✅ PASS FILTERED DATA ONLY
    if plot_prophet(df, metric, fname, mapping[metric]):
        return FileResponse(fname)

    return JSONResponse(status_code=400, content={"error": "Not enough data"})



# ================= DBSCAN VISUALIZATION =================
# ================= DBSCAN VISUALIZATION =================
@app.get("/module3/dbscan")
def dbscan_viz(user_id: str = "All"):
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /module2 first"})

    df = FEATURE_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == user_id]

    if df.empty:
        return JSONResponse(status_code=400, content={"error": "No data for selected user"})

    X_scaled = StandardScaler().fit_transform(df[["heart_rate", "steps", "sleep"]])
    labels = DBSCAN(eps=1.2, min_samples=3).fit_predict(X_scaled)
    df["cluster"] = labels

    normal = df[df["cluster"] != -1]
    outliers = df[df["cluster"] == -1]

    plt.figure(figsize=(10, 6))
    plt.scatter(normal["steps"], normal["heart_rate"], label="Normal", alpha=0.6)
    plt.scatter(outliers["steps"], outliers["heart_rate"], label="Outlier", alpha=0.9)

    plt.xlabel("Steps")
    plt.ylabel("Heart Rate")
    plt.title("DBSCAN Clustering")
    plt.legend()
    plt.grid(True)

    fname = "dbscan.png"
    plt.savefig(fname)
    plt.close()

    return FileResponse(fname)
@app.get("/module3/pca-clusters")
def pca_clusters():
    if not os.path.exists("behavior_clusters.csv"):
        return JSONResponse(status_code=400, content={"error": "Run /module2 first"})

    df = pd.read_csv("behavior_clusters.csv")

    plt.figure(figsize=(10, 6))
    for c in df["cluster"].unique():
        d = df[df["cluster"] == c]
        plt.scatter(d["pca1"], d["pca2"], label=f"Cluster {c}", alpha=0.7)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Behavioral Clustering (PCA + KMeans)")
    plt.legend()
    plt.grid(True)

    fname = "pca_clusters.png"
    plt.savefig(fname)
    plt.close()

    return FileResponse(fname)

@app.get("/module3/raw-vs-rolling/{metric}")
def raw_vs_rolling(metric: str, user_id: str = "All"):
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /module2 first"})

    if metric not in ["heart_rate", "steps", "sleep"]:
        return JSONResponse(status_code=400, content={"error": "Invalid metric"})

    df = FEATURE_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == user_id]

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df[metric], label="Raw", alpha=0.6)
    plt.plot(df["date"], df[f"{metric}_mean_7"], label="Rolling Mean (7)", linewidth=2)

    plt.title(f"{metric} – Raw vs Rolling Feature")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)

    fname = f"{metric}_raw_vs_rolling.png"
    plt.savefig(fname)
    plt.close()

    return FileResponse(fname)


# ================= DISTRIBUTION =================
# ================= DISTRIBUTION =================
@app.get("/module3/distribution/{metric}")
def distribution(metric: str, user_id: str = "All"):
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /module2 first"})

    if metric not in ["heart_rate", "steps", "sleep"]:
        return JSONResponse(status_code=400, content={"error": "Invalid metric"})

    df = FEATURE_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == user_id]

    if df.empty:
        return JSONResponse(status_code=400, content={"error": "No data for selected user"})

    colors_map = {
        "heart_rate": "crimson",
        "steps": "royalblue",
        "sleep": "green"
    }

    plt.figure(figsize=(10, 5))
    plt.hist(df[metric], bins=20, color=colors_map[metric], edgecolor="black")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.title(f"{metric} Distribution")
    plt.grid(True)

    fname = f"{metric}_dist.png"
    plt.savefig(fname)
    plt.close()

    return FileResponse(fname)

# ======================================================
# AI Recommendation Generator
# ======================================================
def ai_generate_recommendation(issue, severity):

    prompt = f"""
    A health monitoring system detected an anomaly.

    Issue: {issue}
    Severity: {severity}

    Provide a short, safe, professional lifestyle recommendation.
    Do not diagnose.
    Maximum 25 words.
    """

    gemini_res = call_gemini_api(prompt)
    if gemini_res:
        return gemini_res

    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("AI error:", e)
        return "Maintain healthy habits and continue monitoring."

@app.get("/module3/anomaly-with-recommendations")
def anomaly_with_recommendations(user_id: str = "All"):
    global ANOMALY_DF, CLEAN_DF

    # ------------------ Validate anomaly data ------------------
    if ANOMALY_DF is None or ANOMALY_DF.empty:
        return {"rows": []}

    df_anom = ANOMALY_DF.copy()
    df_anom["user_id"] = df_anom["user_id"].astype(str)

    # ------------------ User filter ------------------
    if user_id != "All":
        df_anom = df_anom[df_anom["user_id"] == str(user_id)]

    if df_anom.empty:
        return {"rows": []}

    # ------------------ Map metric → issue ------------------
    metric_to_issue = {
        "heart_rate_high": "High heart rate",
        "heart_rate_low": "Low heart rate",
        "low_activity": "Low physical activity",
        "sleep_low": "Low sleep duration",
        "sleep_high": "Excessive sleep duration",
        "dbscan_outlier": "Unusual health pattern"
    }

    df_anom["issue"] = df_anom["metric"].map(metric_to_issue)

    # ------------------ Get severity ------------------
    if os.path.exists("anomaly_report.csv"):
        sev = pd.read_csv("anomaly_report.csv")
        sev["user_id"] = sev["user_id"].astype(str)

        df_anom = df_anom.merge(
            sev[["user_id", "metric", "severity"]],
            on=["user_id", "metric"],
            how="left"
        )
    else:
        df_anom["severity"] = "Medium"

    # ------------------ Build optional context ------------------
    user_stats = ""
    if CLEAN_DF is not None:
        dfx = CLEAN_DF.copy()
        dfx["user_id"] = dfx["user_id"].astype(str)

        if user_id != "All":
            dfx = dfx[dfx["user_id"] == str(user_id)]

        if not dfx.empty:
            user_stats = f"""
            Avg heart rate: {round(dfx['heart_rate'].mean(),1)}
            Avg steps: {round(dfx['steps'].mean(),0)}
            Avg sleep: {round(dfx['sleep'].mean(),1)}
            """

    # ------------------ AI CALL ------------------
    def generate_ai_text(issue, severity):
        prompt = f"""
        A wearable system detected a health anomaly.

        Issue: {issue}
        Severity: {severity}

        User summary:
        {user_stats}

        Provide a short, professional, safe lifestyle recommendation.
        Do not diagnose.
        Maximum 25 words.
        """

        gemini_res = call_gemini_api(prompt)
        if gemini_res:
            return gemini_res

        try:
            res = get_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return res.choices[0].message.content.strip()

        except Exception as e:
            print("AI error:", e)
            return "Maintain healthy habits and monitor trends."

    # Generate recommendations
    df_anom["recommendation"] = df_anom.apply(
        lambda x: generate_ai_text(x["issue"], x["severity"]),
        axis=1
    )

    # ------------------ Sort by severity ------------------
    df_anom["severity_rank"] = df_anom["severity"].map({
        "High": 3, "Medium": 2, "Low": 1
    })

    df_anom = df_anom.sort_values(
        by="severity_rank",
        ascending=False
    ).drop(columns="severity_rank")

    # ------------------ Return ------------------
    return {
        "rows": df_anom[[
            "user_id",
            "date",
            "issue",
            "severity",
            "recommendation"
        ]].to_dict(orient="records")
    }




@app.get("/module3/health-score")
def health_score(user_id: str = "All"):
    if CLEAN_DF is None or CLEAN_DF.empty:
        return {"error": "No data available. Run /preprocess first."}

    df = CLEAN_DF.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_id != "All":
        df = df[df["user_id"] == str(user_id)]

    if df.empty:
        return {"error": "No data for selected user."}

    # ===== DATASET AVERAGES =====
    avg_hr = df["heart_rate"].mean()
    avg_steps = df["steps"].mean()
    avg_sleep = df["sleep"].mean()

    # ===== BENCHMARK BASED SCORING =====
    hr_score = max(0, 100 - abs(avg_hr - 70))
    steps_score = min(100, (avg_steps / 10000) * 100)
    sleep_score = max(0, 100 - abs(avg_sleep - 8) * 12.5)

    health_score = round((hr_score + steps_score + sleep_score) / 3, 1)

    if health_score >= 80:
        status = "Within healthy range"
    elif health_score >= 50:
        status = "Moderate deviation"
    else:
        status = "High deviation"

    return {
        "user_id": user_id,
        "health_score": health_score,
        "status": status,
        "components": {
            "heart_rate": round(hr_score, 1),
            "steps": round(steps_score, 1),
            "sleep": round(sleep_score, 1)
        }
    }


from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

@app.get("/module4/report")
def generate_user_report(user_id: str):
    global CLEAN_DF, RULE_RECOMMENDATIONS

    # ---------- VALIDATION ----------
    if CLEAN_DF is None or CLEAN_DF.empty:
        return {"error": "No data available"}

    df_user = CLEAN_DF[CLEAN_DF["user_id"].astype(str) == str(user_id)]

    if df_user.empty:
        return {"error": "User not found"}

    # ---------- PATH SETUP ----------
    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/user_{user_id}_health_report.pdf"
    donut_path = f"reports/donut_{user_id}.png"

    # ---------- DASHBOARD METRICS ----------
    avg_hr = round(df_user["heart_rate"].mean(), 1)
    avg_steps = round(df_user["steps"].mean(), 0)
    avg_sleep = round(df_user["sleep"].mean(), 1)

    # ---------- HEALTH SCORE CALCULATION ----------
    hr_score = max(0, 100 - abs(avg_hr - 70))
    steps_score = min(100, (avg_steps / 10000) * 100)
    sleep_score = max(0, 100 - abs(avg_sleep - 8) * 12.5)

    health_score = round((hr_score + steps_score + sleep_score) / 3, 1)

    # ---------- CREATE DONUT CHART IMAGE ----------
    plt.figure(figsize=(4, 4))
    plt.pie(
        [hr_score, steps_score, sleep_score],
        labels=["Heart Rate", "Steps", "Sleep"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"Health Score: {health_score}%")
    plt.savefig(donut_path)
    plt.close()

    # ---------- PDF CREATION ----------
    styles = getSampleStyleSheet()
    story = []

    # ---------- TITLE ----------
    story.append(Paragraph("<b>User Health Dashboard Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"User ID: {user_id}", styles["Normal"]))
    story.append(Paragraph(f"Total Records: {len(df_user)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------- METRICS SECTION ----------
    story.append(Paragraph("<b>Health Metrics</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"Average Heart Rate: {avg_hr} BPM", styles["Normal"]))
    story.append(Paragraph(f"Average Steps: {avg_steps}", styles["Normal"]))
    story.append(Paragraph(f"Average Sleep: {avg_sleep} hours", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------- DONUT IMAGE ----------
    story.append(Paragraph("<b>Overall Health Status</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    if os.path.exists(donut_path):
        story.append(Image(donut_path, width=200, height=200))
        story.append(Spacer(1, 12))

    # ---------- HEALTH ISSUES ----------
    story.append(Paragraph("<b>Health Issues & Recommendations</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))

    if RULE_RECOMMENDATIONS is not None:
        df_reco = RULE_RECOMMENDATIONS[
            RULE_RECOMMENDATIONS["user_id"].astype(str) == str(user_id)
        ]

        if df_reco.empty:
            story.append(Paragraph("No significant issues detected.", styles["Normal"]))
        else:
            for _, row in df_reco.iterrows():
                story.append(
                    Paragraph(
                        f"- <b>{row['issue']}</b> ({row['severity']}): {row['recommendation']}",
                        styles["Normal"]
                    )
                )

    # ---------- BUILD PDF ----------
    doc = SimpleDocTemplate(pdf_path)
    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"user_{user_id}_health_report.pdf"
    )
@app.get("/module4/user-dashboard-report")
def user_dashboard_report(user_id: str):
    if CLEAN_DF is None:
        return {"error": "Run preprocessing first"}

    df = FEATURE_DF.copy()
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["user_id"] == str(user_id)].sort_values("date")

    if df.empty:
        return {"error": "User not found"}

    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/user_{user_id}_dashboard.pdf"

    styles = getSampleStyleSheet()
    story = []

    # ---------- TITLE ----------
    story.append(Paragraph("<b>FitPulse – Personalized Health Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"User ID: {user_id}", styles["Normal"]))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # ---------- METRICS ----------
    avg_hr = round(df["heart_rate"].mean(), 1)
    avg_steps = round(df["steps"].mean(), 0)
    avg_sleep = round(df["sleep"].mean(), 1)

    story.append(Paragraph("<b>Health Metrics</b>", styles["Heading2"]))
    story.append(Paragraph(f"Average Heart Rate: {avg_hr} BPM", styles["Normal"]))
    story.append(Paragraph(f"Average Steps: {avg_steps}", styles["Normal"]))
    story.append(Paragraph(f"Average Sleep: {avg_sleep} hours", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------- HEALTH SCORE ----------
    hr_score = max(0, 100 - abs(avg_hr - 70))
    steps_score = min(100, (avg_steps / 10000) * 100)
    sleep_score = max(0, 100 - abs(avg_sleep - 7.5) * 10)

    donut_path = f"reports/donut_{user_id}.png"

    plt.figure(figsize=(4, 4))
    plt.pie(
        [hr_score, steps_score, sleep_score],
        labels=["Heart Rate", "Steps", "Sleep"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Overall Health Score")
    plt.savefig(donut_path)
    plt.close()

    story.append(Paragraph("<b>Health Score Breakdown</b>", styles["Heading2"]))
    story.append(Image(donut_path, width=200, height=200))
    story.append(Spacer(1, 20))

    # ---------- ANOMALIES ----------
    story.append(Paragraph("<b>Detected Issues & Recommendations</b>", styles["Heading2"]))

    if RULE_RECOMMENDATIONS is not None:
        reco = RULE_RECOMMENDATIONS[
            RULE_RECOMMENDATIONS["user_id"].astype(str) == str(user_id)
        ]

        if reco.empty:
            story.append(Paragraph("No significant issues detected.", styles["Normal"]))
        else:
            for _, r in reco.iterrows():
                story.append(Paragraph(
                    f"- <b>{r['issue']}</b> ({r['severity']}): {r['recommendation']}",
                    styles["Normal"]
                ))
    else:
        story.append(Paragraph("No recommendation engine data available.", styles["Normal"]))

    # ---------- RECENT ACTIVITY TABLE (USER-FRIENDLY) ----------
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Recent Activity Summary (Last 7 Days)</b>", styles["Heading2"]))

    display_cols = ["date", "steps", "heart_rate", "sleep"]

    table_df = df[display_cols].tail(7).copy()

    # Rename columns for humans
    table_df.rename(columns={
        "date": "Date",
        "steps": "Steps Walked",
        "heart_rate": "Heart Rate (BPM)",
        "sleep": "Sleep (Hours)"
    }, inplace=True)

    # Round values for clean display
    table_df["Sleep (Hours)"] = table_df["Sleep (Hours)"].round(1)

    # Add simple heart rate status (optional but recommended)
    def hr_status(hr):
        if hr < 60:
            return "Low"
        elif hr > 100:
            return "High"
        return "Normal"

    table_df["HR Status"] = table_df["Heart Rate (BPM)"].apply(hr_status)

    story.append(df_to_table(table_df))


    # ---------- RAW vs ROLLING AVERAGE ----------
    metrics = ["heart_rate", "steps", "sleep"]

    for metric in metrics:
        if metric in df.columns:
            df[f"{metric}_rolling"] = df[metric].rolling(7, min_periods=1).mean()

            plt.figure(figsize=(6, 3))
            plt.plot(df["date"], df[metric], label="Raw", alpha=0.7)
            plt.plot(df["date"], df[f"{metric}_rolling"], label="Rolling Avg (7 days)", linewidth=2)
            plt.title(f"{metric.replace('_', ' ').title()} – Raw vs Rolling Average")
            plt.xlabel("Date")
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
            plt.tight_layout()

            plot_path = f"reports/{metric}_rolling_{user_id}.png"
            plt.savefig(plot_path)
            plt.close()

            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"<b>{metric.replace('_', ' ').title()} Trend</b>",
                styles["Heading2"]
            ))
            story.append(Image(plot_path, width=450, height=220))

    # ---------- DBSCAN CLUSTER ----------
    cluster_cols = ["heart_rate", "steps", "sleep"]

    if all(col in df.columns for col in cluster_cols) and len(df) >= 3:
        X = StandardScaler().fit_transform(df[cluster_cols])
        db = DBSCAN(eps=1.2, min_samples=3)
        df["cluster"] = db.fit_predict(X)

        plt.figure(figsize=(5, 4))
        plt.scatter(
            df["steps"],
            df["heart_rate"],
            c=df["cluster"],
            cmap="tab10",
            alpha=0.7
        )
        plt.xlabel("Steps")
        plt.ylabel("Heart Rate")
        plt.title("DBSCAN Activity Clusters")
        plt.tight_layout()

        cluster_path = f"reports/dbscan_{user_id}.png"
        plt.savefig(cluster_path)
        plt.close()

        story.append(Spacer(1, 16))
        story.append(Paragraph("<b>Activity Clustering (DBSCAN)</b>", styles["Heading2"]))
        story.append(Image(cluster_path, width=420, height=300))

    # ---------- BUILD PDF ----------
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"user_{user_id}_dashboard.pdf"
    )

# ================= DOWNLOAD ANOMALIES =================
@app.get("/download-anomalies")
def download_anomalies():
    if not os.path.exists("anomaly_report.csv"):
        return JSONResponse(status_code=404, content={"error": "No anomalies"})

    return FileResponse(
        "anomaly_report.csv",
        media_type="text/csv",
        filename="anomaly_report.csv"
    )
@app.get("/download-report")
def download_report():
    if CLEAN_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run preprocess first"})

    filename = "fitpulse_dashboard_report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ---------- TITLE ----------
    story.append(Paragraph("<b>FitPulse Health Analytics Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))

    # ---------- OVERVIEW ----------
    df = CLEAN_DF.copy()
    overview_data = [
        ["Metric", "Value"],
        ["Rows Loaded", len(df)],
        ["Users", df["user_id"].nunique()],
        ["Days", df["date"].nunique()],
        ["Avg Heart Rate", round(df["heart_rate"].mean(), 1)],
        ["Start Date", str(df["date"].min().date())],
        ["End Date", str(df["date"].max().date())],
    ]

    overview_table = Table(overview_data)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
    ]))

    story.append(Paragraph("<b>Dataset Overview</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(overview_table)
    story.append(Spacer(1, 20))

    # ---------- SAMPLE DATA ----------
    story.append(Paragraph("<b>Sample Records</b>", styles["Heading2"]))
    sample_table = df_to_table(df)
    sample_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(sample_table)
    story.append(Spacer(1, 20))

    # ---------- ANOMALY SUMMARY ----------
    if os.path.exists("anomaly_report.csv"):
        anom_df = pd.read_csv("anomaly_report.csv")
        story.append(Paragraph("<b>Anomaly Summary</b>", styles["Heading2"]))
        anom_table = df_to_table(anom_df)
        anom_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]))
        story.append(anom_table)
        story.append(Spacer(1, 20))

    # ---------- IMAGES ----------
    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    image_files = (
      glob.glob("prophet_heart_rate_*.png") +
      glob.glob("prophet_steps_*.png") +
      glob.glob("prophet_sleep_*.png") +
      ["dbscan.png", "heart_rate_dist.png", "steps_dist.png", "sleep_dist.png"]
    )


    for img in image_files:
        if os.path.exists(img):
            story.append(Image(img, width=400, height=220))
            story.append(Spacer(1, 12))

    # ---------- BUILD ----------
    # ---------- RECOMMENDATIONS ----------
    if os.path.exists("recommendations.csv"):
        rec_df = pd.read_csv("recommendations.csv")
        story.append(Paragraph("<b>Health Recommendations</b>", styles["Heading2"]))
        rec_table = df_to_table(rec_df)
        rec_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 20))

    doc.build(story)

    return FileResponse(
        filename,
        media_type="application/pdf",
        filename=filename
    )



  # ======================================================
# AI WEIGHT GOAL PLANNER (FAST + STABLE)
# ======================================================

PLAN_CACHE = {}

@app.post("/ai/weight-plan")
def ai_weight_plan(payload: dict = Body(...)):

    try:
        age = payload.get("age")
        height = payload.get("height")
        weight = payload.get("weight")
        activity = payload.get("activity")
        goal = payload.get("goal")
        weeks = payload.get("weeks")

        # ---------- CACHE KEY ----------
        key = f"{age}-{height}-{weight}-{activity}-{goal}-{weeks}"

        if key in PLAN_CACHE:
            return {"plan": PLAN_CACHE[key]}

        # ---------- SHORT PROMPT (FASTER) ----------
        prompt = f"""
        {age}y, {height}cm, {weight}kg.
        {activity}.
        Goal: {goal} in {weeks} weeks.

        Give weekly target, calories, habits.
        Max 80 words.
        """

        # ---------- AI CALL ----------
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 120}
        )

        plan = response.text.strip()

        # ---------- SAVE CACHE ----------
        PLAN_CACHE[key] = plan

        return {"plan": plan}

    except Exception:
        # ---------- SAFE FALLBACK ----------
        return {
            "plan": "Focus on balanced meals, hydration, daily walking, strength training, and consistent sleep. Adjust weekly based on progress."
        }

# ======================================================
# AI WEIGHT / FITNESS PLAN (OPENAI SAFE)
# ======================================================

PLAN_CACHE = {}

@app.post("/ai/weight-plan")
def ai_weight_plan(payload: dict = Body(...)):

    try:
        # ---------- READ INPUT ----------
        age = int(payload.get("age", 0))
        height = float(payload.get("height", 0))
        weight = float(payload.get("weight", 0))
        activity = payload.get("activity", "Unknown")
        goal = payload.get("goal", "Improve fitness")
        weeks = int(payload.get("weeks", 8))

        # ---------- VALIDATION ----------
        if height <= 0 or weight <= 0:
            return {"plan": "Please enter valid height and weight."}

        # ---------- CACHE ----------
        key = f"{age}-{height}-{weight}-{activity}-{goal}-{weeks}"
        if key in PLAN_CACHE:
            return {"plan": PLAN_CACHE[key]}

        # ---------- BMI ----------
        height_m = height / 100
        bmi = round(weight / (height_m * height_m), 1)

        # ---------- PROMPT ----------
        prompt = f"""
        Create a simple fitness plan.

        Age: {age}
        BMI: {bmi}
        Activity: {activity}
        Goal: {goal}
        Duration: {weeks} weeks.

        Provide weekly focus, calories, workout type, daily habits.
        Short.
        """

        # ---------- TRY GEMINI FIRST ----------
        gemini_plan = call_gemini_api(prompt)
        if gemini_plan:
            PLAN_CACHE[key] = gemini_plan
            return {"plan": gemini_plan}

        # ---------- OPENAI ----------
        res = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150
        )

        plan = res.choices[0].message.content.strip()

        # ---------- SAVE ----------
        PLAN_CACHE[key] = plan

        return {"plan": plan}

    except Exception as e:
        print("AI Weight Plan error, using smart fallback:", e)
        fallback_plan = f"""### {weeks}-Week Personalized Fitness Roadmap
• Target: Age {age} | Calculated BMI {bmi} | Goal: {goal}
• Nutrition: Focus on whole foods, lean proteins (1.6g/kg), and balanced complex carbs tailored for a {activity.lower()} lifestyle.
• Exercise: 3–4 weekly resistance sessions combined with 8,000–10,000 daily steps.
• Habits: Prioritize 7–8 hours of restorative sleep, 2.5L daily hydration, and consistent weekly tracking."""
        return {"plan": fallback_plan}


# ======================================================
# OPENAI HEALTH CHATBOT
# ======================================================
@app.post("/module5/chat")
def ai_chat(payload: dict = Body(...)):
    global CLEAN_DF, ANOMALY_DF

    question = payload.get("question", "")
    user_id = str(payload.get("user_id", "All"))

    hr = steps = sleep = "NA"

    if CLEAN_DF is not None:
        df = CLEAN_DF.copy()
        df["user_id"] = df["user_id"].astype(str)

        if user_id != "All":
            df = df[df["user_id"] == user_id]

        if not df.empty:
            hr = round(df["heart_rate"].mean(), 1)
            steps = round(df["steps"].mean(), 0)
            sleep = round(df["sleep"].mean(), 1)

    prompt = f"""
    You are a professional wearable health assistant.

    Heart Rate: {hr}
    Steps: {steps}
    Sleep: {sleep}

    Question: {question}

    Provide safe, helpful, non-diagnostic advice.
    """

    # Try Gemini First
    gemini_reply = call_gemini_api(prompt)
    if gemini_reply:
        return {"answer": gemini_reply}

    try:
        res = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120
        )

        answer = res.choices[0].message.content.strip()

    except Exception as e:
        print("AI Chat error, using health guidance fallback:", e)
        q = question.lower()
        metrics_info = f"Current Metrics: Avg HR {hr} bpm | Steps {steps} | Sleep {sleep} hrs.\n\n"
        if any(w in q for w in ["heart", "pulse", "bpm"]):
            answer = metrics_info + "Cardiovascular Advice: Maintain moderate aerobic activities (e.g. brisk walking, cycling), stay hydrated, and practice stress-reduction techniques. If you experience unexpected resting spikes, consult a physician."
        elif any(w in q for w in ["sleep", "tired", "rest", "fatigue"]):
            answer = metrics_info + "Sleep Recovery Advice: Aim for 7–9 hours nightly. Keep a consistent bedtime routine, avoid screens 1 hour before sleep, and keep your bedroom cool and dark."
        elif any(w in q for w in ["step", "walk", "active", "workout", "exercise"]):
            answer = metrics_info + "Activity Advice: Target 8,000–10,000 daily steps. Incorporate short walking breaks throughout the day to boost metabolic rate and circulation."
        elif any(w in q for w in ["diet", "food", "weight", "eat", "calorie"]):
            answer = metrics_info + "Nutrition Advice: Prioritize whole nutrient-dense foods, adequate lean protein, and proper hydration (2–3L water/day). Track weekly trends for sustainable results."
        else:
            answer = metrics_info + "General Wellness Advice: Consistent sleep, daily movement, and stress management are the cornerstones of positive health metrics. Continue tracking anomalies to spot patterns early."

    return {"answer": answer}



