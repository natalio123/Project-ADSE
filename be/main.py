"""RicaWatch API — FastAPI + model YOLOv8 hasil training pilot v2.
Kelas (urutan sama dengan training): Tomat_Segar, Tomat_Menurun, Cabai."""
import hmac, io, os, threading, time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from ultralytics import YOLO

CLASSES = ["Tomat_Segar", "Tomat_Menurun", "Cabai"]
MENURUN_IDS = [1]                                   # sama dengan notebook

# Letakkan bobot hasil training di backend/models/ (best_pilot_v2.pt atau final_all_data.pt)
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "best_pilot_v2.pt"))
IMGSZ = int(os.getenv("IMGSZ", "960"))              # sama dengan IMGSZ training
CONF = float(os.getenv("CONF", "0.25"))             # sama dengan analyze_photo() di notebook
API_KEY = os.getenv("API_KEY", "")                  # opsional; kosong = tanpa kunci (frontend memanggil langsung)
ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]   # mis. https://ricawatch.vercel.app
ORIGIN_REGEX = os.getenv("ORIGIN_REGEX") or None    # mis. https://ricawatch-.*\.vercel\.app untuk preview deploy
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "300"))  # batas permintaan per IP per menit
MAX_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000

state, lock = {}, threading.Lock()


@asynccontextmanager
async def lifespan(_):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model tidak ditemukan: {MODEL_PATH}. Salin file .pt hasil training ke folder models/.")
    model = YOLO(str(MODEL_PATH))
    names = [model.names[i] for i in sorted(model.names)]
    if names != CLASSES:                            # cegah label tertukar diam-diam
        raise RuntimeError(f"Kelas model {names} != CLASSES {CLASSES}")
    model.predict(Image.new("RGB", (IMGSZ, IMGSZ)), imgsz=IMGSZ, verbose=False)   # warm-up
    state["model"] = model
    yield


app = FastAPI(title="RicaWatch API", lifespan=lifespan, docs_url=None, redoc_url=None)
if ORIGINS or ORIGIN_REGEX:
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_origin_regex=ORIGIN_REGEX,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

hits = defaultdict(deque)


def rate_limit(request: Request):
    """Pembatas sederhana per IP (di belakang Cloudflare Tunnel, IP asli ada di CF-Connecting-IP)."""
    ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
    now, q = time.monotonic(), hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "Terlalu banyak permintaan")
    q.append(now)
    if len(hits) > 5000:                            # cegah memori membengkak
        hits.clear()


def require_key(x_api_key: str = Header(default="")):
    if API_KEY and not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Unauthorized")


@app.get("/health")
def health():
    return {"ok": "model" in state, "model": MODEL_PATH.name}


def analyze(img: Image.Image) -> dict:
    """Setara analyze_photo() di notebook, ditambah kotak deteksi untuk UI."""
    with lock:                                      # model tidak thread-safe
        r = state["model"].predict(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    ids, cnf, boxes = r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()

    counts, mean_conf = {}, {}
    for i, name in enumerate(CLASSES):
        sel = [c for k, c in zip(ids, cnf) if k == i]
        counts[name] = len(sel)
        mean_conf[name] = round(sum(sel) / len(sel), 3) if sel else None

    n_tomat = counts["Tomat_Segar"] + counts["Tomat_Menurun"]
    ada_menurun = any(k in MENURUN_IDS for k in ids)
    detections = [{"label": CLASSES[k], "conf": round(c, 3), "box": [round(v, 4) for v in b]}
                  for k, c, b in zip(ids, cnf, boxes)]

    if not ids:
        commodity, status, confs = "Tidak terdeteksi", "-", []
    elif n_tomat >= counts["Cabai"]:
        commodity = "Tomat"
        status = "Menurun" if ada_menurun else "Segar"
        confs = [c for k, c in zip(ids, cnf) if k != 2]
    else:
        commodity, status = "Cabai", "Terdeteksi"    # model v2: cabai segar+menurun digabung
        confs = [c for k, c in zip(ids, cnf) if k == 2]

    return {
        "commodity": commodity,
        "status": status,
        "confidence": round(sum(confs) / len(confs) * 100, 1) if confs else 0.0,
        "ada_menurun": ada_menurun,
        "menurun_ratio": round(counts["Tomat_Menurun"] / n_tomat, 3) if n_tomat else None,
        "counts": counts,
        "mean_conf": mean_conf,
        "detections": detections,
    }


@app.post("/api/scan", dependencies=[Depends(rate_limit), Depends(require_key)])
def scan(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Gunakan JPEG/PNG/WebP")
    raw = image.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Foto terlalu besar (maks 8 MB)")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Gambar tidak valid")
    return analyze(img)
