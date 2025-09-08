# ThreatLog Detector
Proyecto de Ciberseguridad + IA para detección de anomalías en logs.  
Incluye backend con FastAPI y dashboard con Streamlit.

Plataforma ligera para **ingesta de logs**, **detección de anomalías** y **observabilidad** en tiempo real.  
Incluye API (FastAPI), persistencia en Postgres, y un dashboard interactivo (Streamlit) con métricas, gráficos y un reporte automático en texto (seguridad + rendimiento).

## ✨ Características

- 📥 **Ingesta** de CSV con logs web (endpoint `/ingest`)
- 🤖 **Detección de anomalías** con IsolationForest + reglas heurísticas (5xx / latencias altas)
- 📊 **Métricas** y **gráficas**: errores 5xx, latencias, paths más lentos, distribución de códigos, scatter bytes vs tiempo, etc.
- 🧾 **Reporte automático** (`/report`) con resumen de seguridad y rendimiento
- 🧱 **Persistencia** en **PostgreSQL**
- 🐳 **Despliegue con Docker Compose** (API + Dashboard + DB)
- 🧰 Código limpio con SQLAlchemy (ORM) y FastAPI

## 🧱 Stack

- **API**: FastAPI (Python 3.10+)
- **ML**: scikit-learn (IsolationForest)
- **DB**: PostgreSQL + SQLAlchemy
- **Dashboard**: Streamlit + Plotly
- **Infra**: Docker & Docker Compose

## 🚀 Ejecutar en local

1. Clona el repositorio
2. (Opcional) Copia el ejemplo de variables:
   ```bash
   cp .env.example .env
