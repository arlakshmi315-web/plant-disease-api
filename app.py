from fastapi import FastAPI, UploadFile, Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone

app = FastAPI(title="Plant Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model + supporting files (unchanged from the working version)
# ---------------------------------------------------------------------------
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "plant_disease_model.h5"))

with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

with open(os.path.join(BASE_DIR, "disease_info.json")) as f:
    disease_info = json.load(f)

# ---------------------------------------------------------------------------
# Storage layer
#
# Uses Postgres when DATABASE_URL is set (this is what you MUST configure on
# Render, because Render's own filesystem is wiped on every redeploy and every
# sleep/wake cycle -- SQLite alone will silently lose all scan history).
# Falls back to a local SQLite file only for local development.
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

    # Render's DATABASE_URL sometimes starts with postgres:// - psycopg2 wants postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    def get_conn():
        return psycopg2.connect(DATABASE_URL)

    def init_db():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                latitude REAL,
                longitude REAL
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

    def log_scan(disease, confidence, lat, lon):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO scans (timestamp, disease, confidence, latitude, longitude) "
            "VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(timezone.utc), disease, confidence, lat, lon),
        )
        conn.commit()
        cur.close()
        conn.close()

    def fetch_scans(limit, offset):
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, timestamp, disease, confidence, latitude, longitude "
            "FROM scans ORDER BY timestamp DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]

    def fetch_stats():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM scans")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT disease, COUNT(*) c FROM scans GROUP BY disease ORDER BY c DESC LIMIT 10"
        )
        by_disease = [{"disease": d, "count": c} for d, c in cur.fetchall()]
        cur.execute(
            "SELECT COUNT(*) FROM scans WHERE timestamp > NOW() - INTERVAL '24 hours'"
        )
        last_24h = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"total_scans": total, "scans_last_24h": last_24h, "top_diseases": by_disease}

else:
    DB_PATH = os.path.join(BASE_DIR, "scans.db")

    def get_conn():
        return sqlite3.connect(DB_PATH)

    def init_db():
        conn = get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                latitude REAL,
                longitude REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_scan(disease, confidence, lat, lon):
        conn = get_conn()
        conn.execute(
            "INSERT INTO scans (timestamp, disease, confidence, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), disease, confidence, lat, lon),
        )
        conn.commit()
        conn.close()

    def fetch_scans(limit, offset):
        conn = get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, disease, confidence, latitude, longitude "
            "FROM scans ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fetch_stats():
        conn = get_conn()
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        by_disease = conn.execute(
            "SELECT disease, COUNT(*) c FROM scans GROUP BY disease ORDER BY c DESC LIMIT 10"
        ).fetchall()
        last_24h = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE timestamp > datetime('now', '-1 day')"
        ).fetchone()[0]
        conn.close()
        return {
            "total_scans": total,
            "scans_last_24h": last_24h,
            "top_diseases": [{"disease": d, "count": c} for d, c in by_disease],
        }


init_db()

# ---------------------------------------------------------------------------
# Admin auth (HTTP Basic, credentials from environment variables)
# ---------------------------------------------------------------------------
security = HTTPBasic()

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Admin credentials are not configured on the server.",
        )
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Plant Disease Prediction API is running!", "storage": "postgres" if USE_POSTGRES else "sqlite"}


@app.post("/predict")
async def predict(file: UploadFile, latitude: float = None, longitude: float = None):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(arr)[0]

    # Top-3 predictions, not just the top one -- several classes look similar
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"disease": labels[int(i)], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]

    idx = int(top3_idx[0])
    class_name = labels[idx]
    confidence = float(preds[idx])

    try:
        log_scan(class_name, round(confidence * 100, 2), latitude, longitude)
    except Exception as e:
        # Never let logging failures break a prediction response
        print(f"Warning: failed to log scan: {e}")

    return {
        "disease": class_name,
        "confidence": round(confidence * 100, 2),
        "low_confidence": confidence < 0.9,
        "top3": top3,
        "info": disease_info.get(class_name, {}),
    }


# ---------------------------------------------------------------------------
# Admin endpoints (HTTP Basic auth required)
# ---------------------------------------------------------------------------
@app.get("/admin/scans")
async def admin_scans(
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    username: str = Depends(require_admin),
):
    return {"scans": fetch_scans(limit, offset)}


@app.get("/admin/stats")
async def admin_stats(username: str = Depends(require_admin)):
    return fetch_stats()
