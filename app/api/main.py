# app/api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any, List

from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from sqlalchemy import func, and_

from app.api.db import SessionLocal, LogEntry, init_db


app = FastAPI(title="ThreatLog API", version="0.1.0")

# CORS (ajusta orígenes en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Estado en memoria (opcional para respuestas rápidas) ----
DATA_DF: pd.DataFrame | None = None
METRICS: Dict[str, Any] | None = None

REQUIRED_COLS = [
    "timestamp", "source_ip", "dest_ip", "path",
    "status_code", "bytes", "response_time_ms"
]


# ---- Inicializa DB al arrancar la app ----
@app.on_event("startup")
def on_startup():
    init_db()


# ---------- utilidades de serialización ----------
def to_native(x: Any) -> Any:
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if pd.isna(x): return None
    return x


def df_records_native(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.replace({np.nan: None})
    recs = df2.to_dict(orient="records")
    return [{k: to_native(v) for k, v in r.items()} for r in recs]


# --------- modelo simple de anomalías ----------
def _detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    num_cols = ["status_code", "bytes", "response_time_ms"]
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[num_cols] = work[num_cols].fillna(0)

    # Para datasets pequeños, subimos la contaminación para "ver" puntos rojos en demo
    n = max(1, len(work))
    contamination = 0.03 if n > 50 else 0.15

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    scores = model.fit_predict(work[num_cols])           # -1 anómalo, 1 normal
    decision = model.decision_function(work[num_cols])   # más bajo = más raro

    # Regla heurística adicional (útil para demo)
    rule = (work["status_code"] >= 500) | (work["response_time_ms"] > 250)

    df["anomaly"] = (scores == -1) | rule
    df["anomaly_score"] = -decision
    return df


# --------------- endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"required_columns": REQUIRED_COLS}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Sube CSV, detecta anomalías y persiste en Postgres.
    """
    global DATA_DF, METRICS

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/csv"):
        raise HTTPException(status_code=400, detail="Sube un CSV válido")

    content = await file.read()
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {e}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    # Normalizaciones mínimas
    # ⚠️ Intentamos parsear a datetime; si falla, se descarta esa fila
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()

    for c in ["source_ip", "dest_ip", "path"]:
        df[c] = df[c].astype(str)

    # Detección de anomalías
    df = _detect_anomalies(df)

    # Métricas rápidas en memoria (del lote actual)
    METRICS = {
        "rows": int(df.shape[0]),
        "errors_5xx": int((pd.to_numeric(df["status_code"], errors="coerce") // 100 == 5).sum()),
        "avg_resp_ms": float(pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0).mean()),
    }
    DATA_DF = df.copy()

    # ---- Persistir en Postgres ----
    db = SessionLocal()
    try:
        # Tipado robusto
        df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce").fillna(0).astype(int)
        df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype(int)
        df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0).astype(float)
        df["anomaly"] = df["anomaly"].fillna(False).astype(bool)
        df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0).astype(float)

        # timestamp -> naive datetime (sin tz) para el modelo DateTime por defecto
        def _to_naive_py_dt(x):
            if hasattr(x, "to_pydatetime"):
                x = x.to_pydatetime()
            try:
                if getattr(x, "tzinfo", None) is not None:
                    x = x.replace(tzinfo=None)
            except Exception:
                pass
            return x

        df["timestamp"] = df["timestamp"].apply(_to_naive_py_dt)

        # Inserción en lote
        objs = []
        for r in df.to_dict(orient="records"):
            objs.append(LogEntry(
                timestamp=r.get("timestamp"),
                source_ip=r.get("source_ip", ""),
                dest_ip=r.get("dest_ip", ""),
                path=r.get("path", ""),
                status_code=int(r.get("status_code", 0)),
                bytes=int(r.get("bytes", 0)),
                response_time_ms=float(r.get("response_time_ms", 0.0)),
                anomaly=bool(r.get("anomaly", False)),
                anomaly_score=float(r.get("anomaly_score", 0.0)),
            ))

        if objs:
            db.bulk_save_objects(objs)
            db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()

    return JSONResponse({"status": "ingested", "rows": int(df.shape[0])})


@app.get("/metrics")
def metrics():
    """Métricas agregadas leídas desde la base de datos."""
    db = SessionLocal()
    try:
        rows = db.query(func.count(LogEntry.id)).scalar() or 0
        errors_5xx = (
            db.query(func.count(LogEntry.id))
              .filter(and_(LogEntry.status_code >= 500, LogEntry.status_code < 600))
              .scalar() or 0
        )
        avg_resp = db.query(func.avg(LogEntry.response_time_ms)).scalar()
        avg_resp = float(avg_resp) if avg_resp is not None else 0.0

        return {
            "rows": int(rows),
            "errors_5xx": int(errors_5xx),
            "avg_resp_ms": avg_resp,
        }
    finally:
        db.close()


@app.get("/metrics/advanced")
def metrics_advanced(top: int = 10):
    """
    Devuelve:
      - errors_by_ip: [{source_ip, errors, total, error_pct}]
      - slow_paths:   [{path, avg_ms, total}]
      - latency_series: [{timestamp, response_time_ms}] (últimos N)
    """
    db = SessionLocal()
    try:
        # --- Errores 5xx por IP --------------------------------------------
        totals = dict(
            db.query(LogEntry.source_ip, func.count().label("total"))
              .group_by(LogEntry.source_ip)
              .all()
        )
        errors = dict(
            db.query(LogEntry.source_ip, func.count().label("errors"))
              .filter(and_(LogEntry.status_code >= 500, LogEntry.status_code < 600))
              .group_by(LogEntry.source_ip)
              .all()
        )

        errors_by_ip = []
        for ip, total in totals.items():
            err = errors.get(ip, 0)
            pct = (err / total * 100.0) if total else 0.0
            errors_by_ip.append({
                "source_ip": ip,
                "errors": int(err),
                "total": int(total),
                "error_pct": round(pct, 2),
            })
        errors_by_ip.sort(key=lambda r: r["error_pct"], reverse=True)
        errors_by_ip = errors_by_ip[:top]

        # --- Paths más lentos (promedio ms) --------------------------------
        slow_q = (
            db.query(
                LogEntry.path.label("path"),
                func.avg(LogEntry.response_time_ms).label("avg_ms"),
                func.count().label("total"),
            )
            .group_by(LogEntry.path)
            .order_by(func.avg(LogEntry.response_time_ms).desc())
            .limit(top)
            .all()
        )
        slow_paths = [
            {"path": p, "avg_ms": float(avg or 0.0), "total": int(t)}
            for p, avg, t in slow_q
        ]

        # --- Serie de latencias (últimos 200) -------------------------------
        last_lat = (
            db.query(LogEntry.timestamp, LogEntry.response_time_ms)
              .order_by(LogEntry.id.desc())
              .limit(200)
              .all()
        )
        latency_series = [
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "response_time_ms": float(rt or 0.0),
            }
            for ts, rt in reversed(last_lat)
        ]

        return {
            "errors_by_ip": errors_by_ip,
            "slow_paths": slow_paths,
            "latency_series": latency_series,
        }
    finally:
        db.close()


@app.get("/anomalies")
def anomalies(limit: int = 100):
    """
    Lee anomalías desde Postgres (tabla logs).
    Respuesta compatible con el dashboard (items = lista de dicts).
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(LogEntry)
              .filter(LogEntry.anomaly == True)  # noqa: E712
              .order_by(LogEntry.id.desc())
              .limit(limit)
              .all()
        )
        items = []
        for r in rows:
            items.append({
                "timestamp": r.timestamp,
                "source_ip": r.source_ip,
                "dest_ip": r.dest_ip,
                "path": r.path,
                "status_code": r.status_code,
                "bytes": r.bytes,
                "response_time_ms": r.response_time_ms,
                "anomaly": r.anomaly,
                "anomaly_score": r.anomaly_score,
            })
        return {"items": items}
    finally:
        db.close()


@app.get("/report", response_class=PlainTextResponse)
def report(hours: int = 24, top: int = 5):
    """
    Mini-reporte en texto: seguridad + performance del último X horas.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        total = (
            db.query(func.count(LogEntry.id))
              .filter(LogEntry.timestamp >= cutoff)
              .scalar() or 0
        )
        errors_5xx = (
            db.query(func.count(LogEntry.id))
              .filter(LogEntry.timestamp >= cutoff)
              .filter(and_(LogEntry.status_code >= 500, LogEntry.status_code < 600))
              .scalar() or 0
        )
        avg_ms = (
            db.query(func.avg(LogEntry.response_time_ms))
              .filter(LogEntry.timestamp >= cutoff)
              .scalar()
        )
        avg_ms = float(avg_ms) if avg_ms is not None else 0.0

        # Top IPs con errores 5xx
        top_err_ips = (
            db.query(LogEntry.source_ip, func.count().label("errors"))
              .filter(LogEntry.timestamp >= cutoff)
              .filter(and_(LogEntry.status_code >= 500, LogEntry.status_code < 600))
              .group_by(LogEntry.source_ip)
              .order_by(func.count().desc())
              .limit(top).all()
        )

        # Paths más lentos por promedio
        slow_paths = (
            db.query(LogEntry.path, func.avg(LogEntry.response_time_ms).label("avg_ms"))
              .filter(LogEntry.timestamp >= cutoff)
              .group_by(LogEntry.path)
              .order_by(func.avg(LogEntry.response_time_ms).desc())
              .limit(top).all()
        )

        # Anomalías
        anomalies = (
            db.query(LogEntry)
              .filter(LogEntry.timestamp >= cutoff, LogEntry.anomaly == True)  # noqa: E712
              .order_by(LogEntry.anomaly_score.desc())
              .limit(top).all()
        )

        lines = []
        lines.append(f"🕒 Ventana: últimas {hours}h")
        lines.append(f"• Total eventos: {total}")
        lines.append(f"• Errores 5xx: {errors_5xx}")
        lines.append(f"• Latencia media: {avg_ms:.0f} ms")

        lines.append("\nTop IPs con errores 5xx:")
        if top_err_ips:
            for ip, cnt in top_err_ips:
                lines.append(f"  - {ip}: {cnt}")
        else:
            lines.append("  (sin datos)")

        lines.append("\nPaths más lentos (promedio ms):")
        if slow_paths:
            for p, a in slow_paths:
                lines.append(f"  - {p}: {float(a):.0f} ms")
        else:
            lines.append("  (sin datos)")

        lines.append("\nAnomalías recientes (top score):")
        if anomalies:
            for a in anomalies:
                lines.append(f"  - {a.timestamp} {a.source_ip} → {a.path} ({a.status_code}) score={a.anomaly_score:.4f}")
        else:
            lines.append("  (sin datos)")

        return "\n".join(lines).strip()
    finally:
        db.close()
