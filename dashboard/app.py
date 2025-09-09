# app/dashboard/app.py
import io
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------------------------
# Config de página
# ------------------------------------------------------------------------------
st.set_page_config(page_title="ThreatLog Dashboard", page_icon="🛡️", layout="wide")

API_BASE = "http://api:8000"  # nombre del servicio en docker-compose
REQUIRED_COLS = [
    "timestamp", "source_ip", "dest_ip", "path",
    "status_code", "bytes", "response_time_ms",
]

# ------------------------------------------------------------------------------
# Título y subtítulo superiores
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 8px;">
      <h1 style="margin:0; letter-spacing:.2px">ThreatLog Dashboard</h1>
      <p style="margin:2px 0 0 2px; color:#6b7280; font-size:0.95rem">
        Monitorea, ingiere CSV y visualiza métricas y anomalías en tiempo real.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Tema (claro/oscuro) con toggle y CSS variables
# ------------------------------------------------------------------------------
if "dark" not in st.session_state:
    st.session_state.dark = False

with st.sidebar:
    st.markdown("### Apariencia")
    st.session_state.dark = st.toggle("🌞 / 🌙  Modo oscuro", value=st.session_state.dark)

DARK = st.session_state.dark

# Paletas
palette_light = {
    "--bg": "#ffffff",
    "--text": "#0F172A",
    "--muted": "#64748B",
    "--card": "#F8FAFC",
    "--card-border": "#E2E8F0",
    "--accent": "#2563EB",
    "--success-bg": "rgba(34,197,94,.12)",
    "--success-text": "#16A34A",
}
palette_dark = {
    "--bg": "#0B1220",
    "--text": "#E5E7EB",
    "--muted": "#9CA3AF",
    "--card": "#101826",
    "--card-border": "#1F2937",
    "--accent": "#60A5FA",
    "--success-bg": "rgba(16,185,129,.12)",
    "--success-text": "#34D399",
}
p = palette_dark if DARK else palette_light

st.markdown(
    f"""
    <style>
      :root {{
        --bg: {p["--bg"]};
        --text: {p["--text"]};
        --muted: {p["--muted"]};
        --card: {p["--card"]};
        --card-border: {p["--card-border"]};
        --accent: {p["--accent"]};
        --success-bg: {p["--success-bg"]};
        --success-text: {p["--success-text"]};
      }}
      .stApp, .block-container {{
        background: var(--bg);
        color: var(--text);
      }}
      h1,h2,h3,h4,h5,h6 {{ color: var(--text) !important; }}
      .card {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 14px 16px;
      }}
      .success-chip {{
        background: var(--success-bg);
        color: var(--success-text);
        border-radius: 10px;
        padding: 6px 10px;
        display: inline-block;
        font-weight: 600;
      }}
      .stButton > button {{
        border-radius: 10px;
        background: var(--accent);
        border: 0;
        color: white;
        padding: .6rem 1rem;
      }}
      .st-emotion-cache-ue6h4q, .st-emotion-cache-1wmy9hl {{
        background: var(--card) !important;
        border: 1px solid var(--card-border) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Template de Plotly según tema
PLOTLY_TEMPLATE = "plotly_dark" if DARK else "plotly"

# ------------------------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------------------------
try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    st.markdown(f"<div class='success-chip'>{r.json()}</div>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"API not reachable: {e}")

st.divider()

# ------------------------------------------------------------------------------
# Sidebar: Ingesta de múltiples CSV
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ingesta & opciones")
    with st.expander("Columnas requeridas", expanded=False):
        st.code("\n".join(REQUIRED_COLS), language="text")

    files = st.file_uploader("Arrastra uno o varios CSV", type=["csv"], accept_multiple_files=True)

    if st.button("🚀 Procesar CSV", use_container_width=True):
        if not files:
            st.warning("Selecciona al menos un CSV.")
        else:
            for f in files:
                st.info(f"Ingeriendo: {f.name}")
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files={"file": (f.name, f, "text/csv")},
                        timeout=60
                    )
                    if resp.ok:
                        st.success(resp.json())
                    else:
                        st.error(f"{f.name}: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"{f.name}: error de red {e}")

# ------------------------------------------------------------------------------
# Métricas básicas
# ------------------------------------------------------------------------------
st.subheader("📊 Métricas")

def _safe_int(x, default=0):
    try: return int(x)
    except Exception: return default

def _safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

metrics = None
try:
    r = requests.get(f"{API_BASE}/metrics", timeout=5)
    metrics = r.json() if r.status_code == 200 else None
except Exception:
    metrics = None

if not metrics:
    st.info("Sube datos para ver métricas.")
else:
    rows        = _safe_int(metrics.get("rows", 0))
    errors_5xx  = _safe_int(metrics.get("errors_5xx", metrics.get("errors", 0)))
    avg_resp_ms = _safe_float(metrics.get("avg_resp_ms", metrics.get("avg_response_ms", 0)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas procesadas", f"{rows:,}")
    c2.metric("Errores 5xx", f"{errors_5xx:,}")
    c3.metric("Latencia media (ms)", f"{avg_resp_ms:,.0f}")

st.divider()

# ------------------------------------------------------------------------------
# Métricas avanzadas
# ------------------------------------------------------------------------------
st.subheader("📈 Métricas avanzadas")

adv = {}
try:
    rr = requests.get(f"{API_BASE}/metrics/advanced?top=10", timeout=8)
    if rr.status_code == 200:
        adv = rr.json() or {}
except Exception:
    adv = {}

left, right = st.columns(2)

with left:
    df_ip = pd.DataFrame(adv.get("errors_by_ip", []))
    if not df_ip.empty and {"source_ip", "error_pct"}.issubset(df_ip.columns):
        fig = px.bar(
            df_ip, x="source_ip", y="error_pct",
            hover_data=["errors","total"] if {"errors","total"}.issubset(df_ip.columns) else None,
            title="% de errores 5xx por IP",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay suficientes datos para el gráfico de errores por IP.")

with right:
    df_path = pd.DataFrame(adv.get("slow_paths", []))
    if not df_path.empty and {"path","avg_ms"}.issubset(df_path.columns):
        fig = px.bar(
            df_path, x="path", y="avg_ms",
            hover_data=["total"] if "total" in df_path.columns else None,
            title="Paths más lentos (promedio ms)",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay suficientes datos para el gráfico de paths.")

st.divider()

# ------------------------------------------------------------------
# 🔔 Reporte automático (con selector de ventana y auto-refresh)
# ------------------------------------------------------------------
st.subheader("🔔 Reporte automático")

colA, colB = st.columns([1, 1])
with colA:
    hours = st.slider("Ventana (horas)", min_value=1, max_value=720, value=24, step=1, key="report_hours")
with colB:
    auto = st.checkbox("Auto-actualizar cada 20s", value=False, key="auto_refresh")

# Llamada al backend
txt = ""
try:
    resp = requests.get(f"{API_BASE}/report", params={"hours": int(hours), "top": 5}, timeout=8)
    if resp.ok:
        txt = (resp.text or "").strip()
except Exception:
    txt = ""

if txt:
    st.code(txt, language="markdown")
else:
    st.info("Aún no hay reporte disponible para la ventana seleccionada.")

# Opcional: pequeño “countdown” para que se note el refresco en progreso
if auto:
    with st.empty():
        for sec in range(20, 0, -1):
            st.caption(f"Actualizando en {sec}s…")
            time.sleep(1)
    st.rerun()


# ------------------------------------------------------------------------------
# Anomalías
# ------------------------------------------------------------------------------
def fetch_anomalies(limit=1000) -> pd.DataFrame:
    data = []
    try:
        rr = requests.get(f"{API_BASE}/anomalies", params={"limit": limit}, timeout=6)
        if rr.status_code == 200:
            raw = rr.json() or []
            if isinstance(raw, dict):
                if "items" in raw and isinstance(raw["items"], list):
                    data = raw["items"]
                elif "data" in raw and isinstance(raw["data"], list):
                    data = raw["data"]
                else:
                    data = []
            elif isinstance(raw, list):
                data = raw
    except Exception:
        data = []
    return pd.DataFrame(data)

anom_df = fetch_anomalies(limit=1000)

# ------------------------------------------------------------------------------
# Visualizaciones y tabla de anomalías
# ------------------------------------------------------------------------------
if not anom_df.empty:
    ts_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in anom_df.columns), None)

    if ts_col and anom_df[ts_col].dtype == object:
        with pd.option_context("mode.chained_assignment", None):
            anom_df[ts_col] = pd.to_datetime(anom_df[ts_col], errors="coerce")

    with pd.option_context("mode.chained_assignment", None):
        if "anomaly" in anom_df.columns:
            anom_df["status_badge"] = anom_df["anomaly"].map(lambda v: "🛑 Anomalía" if v else "🟢 OK")
        else:
            anom_df["status_badge"] = "—"

    st.subheader("📉 Visualizaciones de anomalías")
    cA, cB = st.columns([2, 1])

    if ts_col and "response_time_ms" in anom_df.columns:
        with cA:
            df_time = anom_df.sort_values(ts_col)
            color_series = df_time.get("anomaly", pd.Series([False]*len(df_time))).map(
                {True: "Anomalía", False: "Normal"}
            )
            fig = px.line(
                df_time, x=ts_col, y="response_time_ms",
                color=color_series, markers=True,
                title="Latencia (ms) en el tiempo",
                labels={"response_time_ms": "ms", ts_col: "Tiempo", "color": "Clase"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        with cA:
            st.info("No hay datos temporales suficientes para la serie de latencia.")

    with cB:
        if "status_code" in anom_df.columns:
            fig2 = px.histogram(
                anom_df, x="status_code",
                nbins=20, title="Distribución de códigos HTTP",
                labels={"status_code": "Código", "count": "Frecuencia"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay columna 'status_code' para el histograma.")

    if {"bytes", "response_time_ms"}.issubset(anom_df.columns):
        fig3 = px.scatter(
            anom_df, x="bytes", y="response_time_ms",
            color=anom_df.get("anomaly", pd.Series([False]*len(anom_df))).map(
                {True: "Anomalía", False: "Normal"}
            ),
            hover_data=[c for c in ["source_ip", "dest_ip", "path", "status_code"] if c in anom_df.columns],
            title="Bytes vs Response Time",
            labels={"bytes": "Bytes", "response_time_ms": "ms", "color": "Clase"},
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🗃️ Anomalías detectadas")
    show_cols = [
        ts_col if ts_col else "timestamp",
        "source_ip", "dest_ip", "path",
        "status_code", "bytes", "response_time_ms",
        "status_badge", "anomaly_score",
    ]
    table_df = anom_df.copy()
    for c in show_cols:
        if c not in table_df.columns:
            table_df[c] = None

    rename_map = {}
    if ts_col and ts_col != "timestamp":
        rename_map[ts_col] = "timestamp"

    table_df = table_df[show_cols].rename(columns={
        **rename_map,
        "status_badge": "estado",
        "anomaly_score": "score",
    })
    table_df["score"] = table_df["score"].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "")
    table_df["response_time_ms"] = table_df["response_time_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "")

    st.dataframe(table_df, use_container_width=True, height=380)

    buff = io.StringIO()
    table_df.to_csv(buff, index=False, encoding="utf-8-sig")
    st.download_button(
        "💾 Descargar anomalías (CSV)",
        data=buff.getvalue().encode("utf-8-sig"),
        file_name="anomalies_export.csv",
        mime="text/csv"
    )
else:
    st.info("Aún no hay anomalías registradas.")


