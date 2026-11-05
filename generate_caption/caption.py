"""CNN captioner inference helpers and command-line generation."""

import argparse
import csv
from datetime import datetime
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_captioner import ImageCaptioner, beam_search, image_transform


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "cnn-transformer" / "model.pt"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_caption_model(checkpoint=DEFAULT_CHECKPOINT, device="auto"):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    saved = torch.load(checkpoint, map_location="cpu")
    config = saved["config"]
    model = ImageCaptioner(
        len(config["vocabulary"]), config["backbone"], config["decoder"],
        pretrained=False, d_model=config["d_model"],
        decoder_layers=config["decoder_layers"],
        attention_heads=config["attention_heads"], max_length=config["max_length"],
    )
    model.load_state_dict(saved["model_state"])
    runtime_device = choose_device(device)
    model.to(runtime_device).eval()
    transform = image_transform(config.get("image_size", 224))
    return model, config["vocabulary"], transform, runtime_device, config


def predict_caption(image, model, vocabulary, id_to_word, device, beam_size=3, max_words=14):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    else:
        image = image.convert("RGB")
    transform = image_transform()
    tensor = transform(image)
    return beam_search(
        model, tensor, vocabulary, id_to_word, device,
        beam_size=beam_size, min_words=10, max_words=max_words,
    )


def parse_plain_captions(response):
    """Parse two plain text caption lines and enforce the dataset word limit."""
    response = re.sub(r"^```(?:text)?\s*|\s*```$", "", response.strip(), flags=re.I)
    captions = []
    for line in response.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        line = line.strip().strip('"“”')
        if line:
            captions.append(line)
    if len(captions) != 2:
        raise ValueError(f"Gemini harus mengembalikan dua baris caption: {response[:180]}")
    for caption in captions:
        count = len(caption.split())
        if not 10 <= count <= 15:
            raise ValueError(f"Caption harus 10–15 kata, ditemukan {count}: {caption}")
    return captions


def generate_gemini_caption(client, model_name, image_path, category):
    """Generate two Indonesian draft captions from one image using Gemini."""
    try:
        from google.genai import types
    except ImportError as error:
        raise SystemExit(
            "Paket Gemini belum terpasang. Jalankan: python -m pip install google-genai"
        ) from error
    prompt = (
        "Buat tepat dua caption berbeda dalam Bahasa Indonesia untuk gambar ini. "
        f"Kategori dataset: {category}. Setiap caption satu kalimat, 10 sampai 15 kata. "
        "Setiap caption wajib mengaitkan gambar dengan Bali: sebut kata Bali atau unsur "
        "budaya Bali yang sesuai kategori dan didukung gambar. Jelaskan hanya hal yang "
        "tampak. Jangan menebak lokasi, nama ritual, makna, atau bahan. Jawab hanya dengan dua "
        "baris teks biasa, satu caption per baris. Jangan gunakan JSON, nomor, label, atau markdown. "
        "Identifikasi kegiatan yang terlihat dan jangan mengarang unsur Bali."
    )
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=160),
        )
    if not response.text:
        raise ValueError("Gemini tidak mengembalikan teks caption.")
    return parse_plain_captions(response.text)


def generate_annotations(annotations_path, model_name, limit=None, overwrite=False):
    """Fill blank annotation rows with Gemini and save progress after each image."""
    try:
        from google import genai
    except ImportError as error:
        raise SystemExit(
            "Paket Gemini belum terpasang. Jalankan: python -m pip install google-genai"
        ) from error
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY belum disetel. Buat API key di Google AI Studio, "
            "lalu ekspor variabel itu di terminal."
        )

    with annotations_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"image_path", "caption", "category", "reviewed"}
    if not required.issubset(fields):
        raise SystemExit(f"Kolom CSV wajib: {', '.join(sorted(required))}")

    if overwrite:
        backup_dir = annotations_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{annotations_path.stem}_before_regenerate_{stamp}.csv"
        shutil.copy2(annotations_path, backup_path)
        for row in rows:
            row["caption"] = ""
            row["reviewed"] = "no"
        temporary_path = annotations_path.with_suffix(annotations_path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8-sig") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(annotations_path)
        print(f"Caption lama dicadangkan ke: {backup_path.relative_to(PROJECT_ROOT)}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["image_path"], []).append(row)
    todo = [path for path, image_rows in grouped.items()
            if any(not row["caption"].strip() for row in image_rows)]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("Tidak ada caption kosong untuk dibuat.")
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"Model: {model_name} | Gambar yang akan diproses: {len(todo)}")
    for index, relative_path in enumerate(todo, start=1):
        image_path = Path(relative_path)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_rows = grouped[relative_path]
        category = image_rows[0]["category"]
        try:
            captions = generate_gemini_caption(client, model_name, image_path, category)
        except Exception as error:
            print(f"[{index}/{len(todo)}] GAGAL {relative_path}: {error}")
            if any(marker in str(error).upper() for marker in ("429", "QUOTA", "RESOURCE_EXHAUSTED")):
                print("Proses dihentikan karena kuota/rate limit. Jalankan ulang untuk melanjutkan.")
                break
            continue
        blank_rows = [row for row in image_rows if not row["caption"].strip()]
        for row, caption in zip(blank_rows, captions):
            row["caption"] = caption
            # Leave human review status unchanged; generated text is a draft.
        temporary_path = annotations_path.with_suffix(annotations_path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8-sig") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(annotations_path)
        print(f"[{index}/{len(todo)}] {relative_path}")
        print(f"Caption 1: {captions[0]}")
        print(f"Caption 2: {captions[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Buat caption satu gambar dengan CNN lokal, atau isi CSV caption menggunakan Gemini API."
    )
    parser.add_argument("image", nargs="?", help="Path satu gambar untuk CNN yang sudah dilatih")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--annotations", help="CSV anotasi untuk mengisi caption kosong dengan Gemini")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--limit", type=int, help="Batas gambar untuk menjalankan percobaan kecil")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Kosongkan dan buat ulang semua caption; CSV lama dicadangkan dahulu.",
    )
    args = parser.parse_args()
    if args.annotations:
        annotations_path = Path(args.annotations)
        if not annotations_path.is_absolute():
            annotations_path = PROJECT_ROOT / annotations_path
        if not annotations_path.is_file():
            raise SystemExit(f"CSV anotasi tidak ditemukan: {annotations_path}")
        generate_annotations(annotations_path, args.gemini_model, args.limit, args.overwrite)
        return
    if not args.image:
        parser.error("berikan path gambar atau gunakan --annotations untuk membuat caption dataset")
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint /= "model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint belum ditemukan: {checkpoint}. Jalankan train_captioner.py dahulu.")

    model, vocabulary, _, device, config = load_caption_model(checkpoint, args.device)
    id_to_word = {str(index): word for word, index in vocabulary.items()}
    print(f"Perangkat: {device}")
    print(predict_caption(args.image, model, vocabulary, id_to_word, device,
                          beam_size=args.beam_size, max_words=config["max_words"]))


if __name__ == "__main__":
    main()
