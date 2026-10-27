from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from generate_caption.caption import load_caption_model, predict_caption


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "outputs" / "blip-finetuned"
MODEL_DIR = Path(os.getenv("CAPTION_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
DEVICE = os.getenv("CAPTION_DEVICE", "auto")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"Model fine-tuned tidak ditemukan di {MODEL_DIR}. "
            "Jalankan train_captioner.py terlebih dahulu."
        )
    model, processor, device = load_caption_model(MODEL_DIR, DEVICE, local_files_only=True)
    app.state.model = model
    app.state.processor = processor
    app.state.device = device
    yield


app = FastAPI(
    title="Bali Image Captioning API",
    description="API caption gambar memakai BLIP yang telah di-fine-tune secara lokal.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_DIR)}


@app.post("/caption")
def create_caption(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Input harus berupa file gambar.")
    try:
        image_data = image.file.read()
        with Image.open(BytesIO(image_data)) as opened:
            input_image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="File tidak dapat dibaca sebagai gambar.")

    caption = predict_caption(
        input_image,
        app.state.model,
        app.state.processor,
        app.state.device,
    )
    return {
        "caption": caption,
        "model": "BLIP fine-tuned",
        "device": str(app.state.device),
    }
