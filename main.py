import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import requests
import io
import warnings
warnings.filterwarnings("ignore")

from explain import ( # pyright: ignore[reportMissingImports]
    distribution_grid, correlation_heatmap,
    missing_values_chart, target_distribution_chart,
)

API_BASE = "https://automl-analyst-backend.onrender.com/"

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        res = getattr(requests, method)(url, **kwargs)
        res.raise_for_status()
        return res
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the FastAPI backend. Run: `uvicorn api.serve:app --reload --port 8000`")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"❌ API error {e.response.status_code}: {detail or str(e)}")
        st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1300px; }
section[data-testid="stSidebar"] { background: #080a0e !important; }
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

.hero {
    background: linear-gradient(135deg, #080a0e 0%, #0e1016 50%, #080a0e 100%);
    border: 1px solid #1a1d2e; border-radius: 20px;
    padding: 36px 40px 30px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.hero-title {
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #e2e8f0 0%, #818cf8 60%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin: 0; line-height: 1.6; }
.hero-pills { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.pill {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
    color: #818cf8; font-size: 11px; font-family: 'JetBrains Mono', monospace;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.5px;
}
.sec-head {
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid #1a1d2e; padding-bottom: 12px; margin: 32px 0 20px;
}
.sec-num {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: white; flex-shrink: 0;
}
.sec-title { font-size: 15px; font-weight: 600; color: #e2e8f0; }
.metric-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }
.mc {
    flex: 1; min-width: 110px; max-width: 180px;
    background: #0d0f16; border: 1px solid #1a1d2e;
    border-radius: 12px; padding: 16px 18px;
    position: relative; overflow: hidden;
}
.mc::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--mc-color, #6366f1); }
.mc-label { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.mc-value { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.mc-sub   { font-size: 10px; color: #6366f1; margin-top: 5px; }
.rt { width: 100%; border-collapse: collapse; }
.rt th { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #475569; padding: 8px 14px; border-bottom: 1px solid #1a1d2e; text-align: left; }
.rt td { padding: 10px 14px; font-size: 13px; border-bottom: 1px solid #0d0f16; font-family: 'JetBrains Mono', monospace; }
.rt tr:hover td { background: #0d0f16; }
.rt-best td { color: #818cf8 !important; font-weight: 600; }
.sb-wrap { background: #1a1d2e; border-radius: 4px; height: 5px; width: 90px; }
.sb { background: linear-gradient(90deg,#6366f1,#818cf8); height: 5px; border-radius: 4px; }
.insight {
    background: #0d0f16; border: 1px solid #1a1d2e;
    border-left: 3px solid #6366f1; border-radius: 0 12px 12px 0;
    padding: 16px 20px; font-size: 14px; color: #94a3b8; line-height: 1.75; margin: 14px 0;
}
.callout {
    background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0;
    font-size: 13px; color: #94a3b8;
}
div.stButton > button { font-weight: 600 !important; border-radius: 8px !important; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    border: none !important; color: white !important;
}
.stProgress > div > div { background: linear-gradient(90deg,#6366f1,#818cf8) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px;'>
      <div style='font-size:1.6rem; margin-bottom:4px;'>⚡</div>
      <div style='font-size:16px; font-weight:700; color:#e2e8f0;'>AutoML Platform</div>
      <div style='font-size:11px; color:#475569; margin-top:2px;'>v2.0 · FastAPI powered</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#475569;margin-bottom:12px;">⚙️ Training settings</div>', unsafe_allow_html=True)
    test_size  = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    cv_folds   = st.slider("Cross-validation folds", 2, 10, 5)
    scaler_opt = st.selectbox("Feature scaler", ["standard", "minmax", "robust"])

    st.markdown("---")
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#475569;margin-bottom:12px;">🔍 Display settings</div>', unsafe_allow_html=True)
    top_n_feat = st.slider("Top N features to show", 5, 25, 15)
    show_all   = st.checkbox("Show all models in table", value=True)

    st.markdown("---")
    # Live API status
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            st.markdown("<div style='font-size:12px;color:#34d399;'>🟢 API connected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:12px;color:#f472b6;'>🔴 API error</div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='font-size:12px;color:#fbbf24;'>🟡 API offline — start it first</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">⚡ AutoML Data Analyst Platform</div>
  <div class="hero-sub">
    Upload any dataset → automatic EDA → feature engineering → train &amp; compare 9+ ML models →
    explainability charts → AI-powered insights. No manual model selection needed.
  </div>
  <div class="hero-pills">
    <span class="pill">pandas</span>
    <span class="pill">scikit-learn</span>
    <span class="pill">matplotlib</span>
    <span class="pill">streamlit</span>
    <span class="pill">fastapi</span>
    <span class="pill">claude-api</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">1</div>
  <div class="sec-title">Upload Dataset</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your CSV or Excel file here",
    type=["csv", "xlsx", "xls"],
    help="Supported: .csv, .xlsx, .xls — up to ~200MB",
)

if not uploaded:
    st.markdown("""
    <div class="callout">
    👆 Upload a dataset to get started.<br>
    Try: <b>Titanic</b> (classification) · <b>House Prices</b> (regression) ·
    <b>Iris</b> (multi-class) · Any tabular dataset from Kaggle.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Send to API once per file ──────────────────────────
file_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("upload_key") != file_key:
    with st.spinner("📤 Uploading to API..."):
        res = api("post", "/upload", files={"file": (uploaded.name, uploaded.getvalue())})
        data = res.json()
        st.session_state["session_id"] = data["session_id"]
        st.session_state["profile"]    = data["profile"]
        st.session_state["columns"]    = data["columns"]
        st.session_state["upload_key"] = file_key
        st.session_state.pop("train_result", None)  # clear stale results

session_id = st.session_state["session_id"]
profile    = st.session_state["profile"]
columns    = st.session_state["columns"]

# ── Summary cards ──
null_avg   = profile["null_pct_avg"]
null_color = "#f472b6" if null_avg > 20 else ("#fbbf24" if null_avg > 5 else "#34d399")
dup_color  = "#f472b6" if profile["duplicates"] > 0 else "#34d399"

st.markdown(f"""
<div class="metric-grid">
  <div class="mc" style="--mc-color:#6366f1"><div class="mc-label">Rows</div><div class="mc-value">{profile['rows']:,}</div></div>
  <div class="mc" style="--mc-color:#818cf8"><div class="mc-label">Columns</div><div class="mc-value">{profile['cols']}</div></div>
  <div class="mc" style="--mc-color:#38bdf8">
    <div class="mc-label">Numeric</div><div class="mc-value">{len(profile['numeric_cols'])}</div>
    <div class="mc-sub">{', '.join(profile['numeric_cols'][:3])}{'...' if len(profile['numeric_cols'])>3 else ''}</div>
  </div>
  <div class="mc" style="--mc-color:#34d399">
    <div class="mc-label">Categorical</div><div class="mc-value">{len(profile['cat_cols'])}</div>
    <div class="mc-sub">{', '.join(profile['cat_cols'][:3])}{'...' if len(profile['cat_cols'])>3 else ''}</div>
  </div>
  <div class="mc" style="--mc-color:{null_color}">
    <div class="mc-label">Avg Null %</div><div class="mc-value">{null_avg}%</div>
    <div class="mc-sub">{"⚠ needs imputation" if null_avg > 5 else "✓ clean"}</div>
  </div>
  <div class="mc" style="--mc-color:{dup_color}">
    <div class="mc-label">Duplicates</div><div class="mc-value">{profile['duplicates']}</div>
    <div class="mc-sub">{"⚠ found" if profile['duplicates']>0 else "✓ none"}</div>
  </div>
  <div class="mc" style="--mc-color:#6366f1">
    <div class="mc-label">Memory</div><div class="mc-value">{profile['memory_mb']}</div>
    <div class="mc-sub">MB</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Lightweight local df just for EDA charts
df_preview = (pd.read_csv(io.BytesIO(uploaded.getvalue()))
              if uploaded.name.endswith(".csv")
              else pd.read_excel(io.BytesIO(uploaded.getvalue())))

with st.expander("👁️ Preview data (first 20 rows)", expanded=False):
    st.dataframe(df_preview.head(20), use_container_width=True)

with st.expander("📊 Column profiles", expanded=False):
    st.dataframe(pd.DataFrame(profile["col_profiles"]), use_container_width=True)


# ══════════════════════════════════════════════════════
#  STEP 2 — EDA  (local charts, no heavy compute)
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">2</div>
  <div class="sec-title">Exploratory Data Analysis</div>
</div>
""", unsafe_allow_html=True)

eda_t1, eda_t2, eda_t3 = st.tabs(["📊 Distributions", "🔗 Correlations", "❓ Missing Values"])

with eda_t1:
    fig = distribution_grid(df_preview, profile["numeric_cols"], max_cols=4)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("No numeric columns found.")

with eda_t2:
    fig = correlation_heatmap(df_preview)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Need at least 2 numeric columns.")

with eda_t3:
    fig = missing_values_chart(df_preview)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown("""
        <div class="insight">
        🔧 <b>Auto-handling:</b> Numeric nulls → imputed with <b>column median</b>.
        Categorical nulls → imputed with <b>most frequent value</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ No missing values — your dataset is clean!")


# ══════════════════════════════════════════════════════
#  STEP 3 — CONFIGURE
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">3</div>
  <div class="sec-title">Configure AutoML</div>
</div>
""", unsafe_allow_html=True)

col_cfg1, col_cfg2 = st.columns([2, 1])
with col_cfg1:
    target = st.selectbox("🎯 Target column (what you want to predict)", columns)
with col_cfg2:
    task = st.radio("📌 Task type", ["classification", "regression"], horizontal=True)

if target:
    val_res = api("get", f"/dataset/{session_id}/validate-target", params={"target": target})
    val     = val_res.json()
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown(f"""
        <div class="callout">
        <b>Target: {target}</b><br>
        Unique values: <b>{val['n_unique']}</b><br>
        Null rows: <b>{val['null_count']}</b> ({val['null_pct']}%)<br>
        Auto-detected task: <b>{val['task_hint']}</b>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        fig = target_distribution_chart(df_preview[target], task)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

run_btn = st.button("🚀 Run AutoML — Train All Models", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════
#  STEP 4 — TRAIN  (all heavy work done by FastAPI)
# ══════════════════════════════════════════════════════
if run_btn:
    with st.spinner("🤖 FastAPI is training all models... this may take a minute."):
        res = api(
            "post", f"/train/{session_id}",
            data={
                "target":    target,
                "task":      task,
                "test_size": test_size,
                "cv_folds":  cv_folds,
                "scaler":    scaler_opt,
            }
        )
        st.session_state["train_result"] = res.json()

if "train_result" not in st.session_state:
    st.stop()

result      = st.session_state["train_result"]
leaderboard = result["leaderboard"]
best        = result["best_model"]
task        = result["task"]
prep        = result["preprocessing"]

st.markdown("""
<div class="sec-head">
  <div class="sec-num">4</div>
  <div class="sec-title">AutoML Results</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="callout">
✅ Preprocessing complete &nbsp;·&nbsp;
Original features: <b>{prep['original_features']}</b> →
Transformed features: <b>{prep['transformed_features']}</b> &nbsp;·&nbsp;
Train rows: <b>{prep['train_rows']:,}</b> &nbsp;·&nbsp;
Test rows: <b>{prep['test_rows']:,}</b>
</div>
""", unsafe_allow_html=True)

# ── Best model banner ──
score_key   = "accuracy" if task == "classification" else "r2"
score_label = "Accuracy" if task == "classification" else "R²"
best_score  = best.get(score_key, 0)
cv_mean     = best.get("cv_mean", 0)
cv_std      = best.get("cv_std", 0)

st.markdown(f"""
<div class="metric-grid">
  <div class="mc" style="--mc-color:#6366f1">
    <div class="mc-label">🏆 Best Model</div>
    <div class="mc-value" style="font-size:1rem;padding-top:4px">{best['model']}</div>
  </div>
  <div class="mc" style="--mc-color:#34d399">
    <div class="mc-label">{score_label}</div>
    <div class="mc-value">{best_score:.4f}</div>
    <div class="mc-sub">Test set</div>
  </div>
  <div class="mc" style="--mc-color:#38bdf8">
    <div class="mc-label">CV Score ({cv_folds}-fold)</div>
    <div class="mc-value">{cv_mean:.4f}</div>
    <div class="mc-sub">±{cv_std:.4f}</div>
  </div>
  <div class="mc" style="--mc-color:#818cf8">
    <div class="mc-label">Models Tested</div>
    <div class="mc-value">{result['models_tested']}</div>
  </div>
  <div class="mc" style="--mc-color:#fbbf24">
    <div class="mc-label">Train Time</div>
    <div class="mc-value">{best.get('train_time', 0)}s</div>
    <div class="mc-sub">best model</div>
  </div>
  <div class="mc" style="--mc-color:#f472b6">
    <div class="mc-label">Features</div>
    <div class="mc-value">{prep['transformed_features']}</div>
    <div class="mc-sub">after encoding</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Leaderboard ──
df_lb     = pd.DataFrame(leaderboard)
cols_show = df_lb.columns.tolist()
max_s     = df_lb[score_key].max()
min_s     = df_lb[score_key].min()
rng       = max(max_s - min_s, 1e-9)

rows_html = ""
for i, row in df_lb.iterrows():
    crown   = "👑 " if i == 0 else "&nbsp;&nbsp;&nbsp;"
    cls     = "rt-best" if i == 0 else ""
    bar_pct = int((row[score_key] - min_s) / rng * 100)
    cells   = f"<td>{crown}{row['model']}</td>"
    for col in cols_show[1:]:
        cells += f"<td>{row.get(col, '—')}</td>"
    cells += f"<td><div class='sb-wrap'><div class='sb' style='width:{bar_pct}%'></div></div></td>"
    rows_html += f"<tr class='{cls}'>{cells}</tr>"

headers = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in cols_show) + "<th>Bar</th>"
st.markdown(f"""
<div style="overflow-x:auto;margin:20px 0;">
<table class="rt"><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  STEP 5 — ANALYSIS
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">5</div>
  <div class="sec-title">Model Analysis &amp; Explainability</div>
</div>
""", unsafe_allow_html=True)

tab_feat, tab_cmp = st.tabs(["🎯 Feature Importance", "📊 Model Comparison"])

with tab_feat:
    try:
        fi_res  = api("get", f"/results/{session_id}/feature-importance", params={"top_n": top_n_feat})
        fi_data = fi_res.json()
    except Exception:
        fi_data = {}
    if fi_data.get("features"):
        fi_df = pd.DataFrame(fi_data["features"])
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0b0d12"); ax.set_facecolor("#0f1117")
        ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#6366f1")
        ax.set_xlabel("Importance", color="#94a3b8")
        ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2234")
        ax.set_title(f"Top {top_n_feat} Features — {fi_data['model']}", color="#e2e8f0")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Feature importance not available for this model type.")

with tab_cmp:
    metric_options = (
        ["accuracy", "f1_score", "cv_mean"] if task == "classification"
        else ["r2", "rmse", "mae", "cv_mean"]
    )
    metric_sel = st.selectbox("Compare models by", metric_options)
    if metric_sel in df_lb.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0b0d12"); ax.set_facecolor("#0f1117")
        colors = ["#6366f1" if i == 0 else "#1a1d2e" for i in range(len(df_lb))]
        ax.barh(df_lb["model"][::-1], df_lb[metric_sel][::-1], color=colors[::-1])
        ax.set_xlabel(metric_sel.replace("_", " ").title(), color="#94a3b8")
        ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2234")
        ax.set_title(f"Model Comparison — {metric_sel}", color="#e2e8f0")
        st.pyplot(fig, use_container_width=True); plt.close(fig)


# ══════════════════════════════════════════════════════
#  STEP 6 — INSIGHTS
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">6</div>
  <div class="sec-title">Insights &amp; Interpretation</div>
</div>
""", unsafe_allow_html=True)

cv_gap = abs(best_score - cv_mean)
overfit = " ⚠️ Possible overfitting — CV score is notably lower." if cv_gap > 0.08 else ""

if task == "classification":
    insight = (
        f"✅ <b>{best['model']}</b> achieved <b>{best_score:.2%} test accuracy</b> "
        f"and <b>{cv_mean:.2%} cross-validated accuracy</b> ({cv_folds} folds).{overfit} "
        f"<br><br>💡 <b>Next steps:</b> Tune hyperparameters with GridSearchCV, "
        f"engineer interaction features, and check for class imbalance."
    )
else:
    insight = (
        f"✅ <b>{best['model']}</b> explains <b>{best_score*100:.1f}% of variance (R²={best_score:.4f})</b> "
        f"in <i>{target}</i>, with CV R² of <b>{cv_mean:.4f}</b>.{overfit} "
        f"<br><br>💡 <b>Next steps:</b> Inspect residuals, apply log-transform to skewed features, "
        f"and tune {best['model']} with GridSearchCV."
    )

st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  STEP 7 — EXPORT
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
  <div class="sec-num">7</div>
  <div class="sec-title">Export Results</div>
</div>
""", unsafe_allow_html=True)

col_x1, col_x2, col_x3 = st.columns(3)

with col_x1:
    st.download_button(
        "⬇️ Leaderboard CSV",
        df_lb.to_csv(index=False),
        "automl_leaderboard.csv", "text/csv",
        use_container_width=True,
    )

with col_x2:
    pred_bytes = api("get", f"/export/{session_id}/predictions").content
    st.download_button(
        "⬇️ Predictions CSV", pred_bytes,
        "predictions.csv", "text/csv",
        use_container_width=True,
    )

with col_x3:
    try:
        fi_all = api("get", f"/results/{session_id}/feature-importance", params={"top_n": 100}).json()
    except Exception:
        fi_all = {}
    if fi_all.get("features"):
        fi_csv = pd.DataFrame(fi_all["features"]).to_csv(index=False)
        st.download_button(
            "⬇️ Feature Importances CSV", fi_csv,
            "feature_importances.csv", "text/csv",
            use_container_width=True,
        )
    else:
        st.button("⬇️ Feature Importances CSV", disabled=True, use_container_width=True)

model_name = best["model"].replace(" ", "_")
pkl_bytes  = api("get", f"/export/{session_id}/model").content
st.download_button(
    f"⬇️ Download Trained Model — {best['model']} (.pkl)",
    pkl_bytes,
    f"{model_name}.pkl",
    "application/octet-stream",
    use_container_width=True,
)
 
st.markdown(f"""
<div class="callout" style="margin-top:16px">
✅ Training complete &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
