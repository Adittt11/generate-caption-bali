import csv
import mimetypes
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image


# Lokasi dibuat relatif terhadap file ini supaya skrip bisa dijalankan
# dari folder proyek maupun dari dalam folder generate_caption.
PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "scraping_image" / "kerajinan_bali"
CSV_PATH = Path(__file__).resolve().parent / "kerajinan_bali_gemini_2caption.csv"
ENV_PATH = Path(__file__).resolve().parent / ".env"

MODEL_ID = "gemini-3.5-flash-lite"
MIN_WORDS = 20
MAX_WORDS = 35
PAUSE_BETWEEN_REQUESTS = 3
MAX_ATTEMPTS = 5

PROMPT = """
Buat tepat dua caption berbeda dalam Bahasa Indonesia untuk gambar kerajinan
Bali ini. Masing-masing harus 20 sampai 35 kata. Jelaskan hanya hal yang
terlihat, seperti bentuk, warna, pola, tekstur, bahan yang dapat dikenali,
detail hiasan, atau tahap pembuatan jika ada orang yang sedang bekerja.
Sebutkan nama benda, bahan, atau teknik hanya jika cirinya terlihat jelas.
Jangan menebak fungsi, asal, sejarah, atau makna simbol. Gunakan fokus atau
susunan kalimat berbeda untuk tiap caption. Hindari pembukaan
'Gambar ini menunjukkan'.

Keluarkan hanya dua baris dengan format:
1. [caption pertama]
2. [caption kedua]
""".strip()


def load_env_file():
    """Muat key-value sederhana dari file .env di samping skrip."""
    if not ENV_PATH.is_file():
        return

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name:
            # Nilai dari .env dipakai untuk skrip ini jika file sudah diisi.
            os.environ[name] = value


def count_words(text):
    return len(text.split()) if text else 0


def parse_caption_pair(text):
    captions = {}
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*[-*`#]+\s*|\s*[*`]+\s*$", "", line).strip()
        match = re.match(
            r"^\s*(?:caption\s*)?(1|2|pertama|kedua)\s*[.):\-]\s*(.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if match:
            label = match.group(1).lower()
            number = {"1": 1, "pertama": 1, "2": 2, "kedua": 2}[label]
            captions[number] = match.group(2).strip().strip("*` ")

    if set(captions) != {1, 2}:
        return None
    first, second = captions[1], captions[2]
    if not all(MIN_WORDS <= count_words(c) <= MAX_WORDS for c in (first, second)):
        return None
    if first.casefold() == second.casefold():
        return None
    return first, second


def is_retryable(error):
    message = str(error).lower()
    return any(marker in message for marker in ("429", "resource_exhausted", "503", "500", "unavailable"))


def generate_caption(client, image_path):
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Format gambar tidak dikenali: {image_path.name}")

    image_bytes = image_path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    invalid_response_attempts = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prompt = PROMPT
            if invalid_response_attempts:
                prompt += (
                    "\nPerbaikan: keluarkan tepat dua baris berlabel `1.` dan `2.`, "
                    "dan pastikan masing-masing caption berisi 20-35 kata."
                )
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(temperature=0.4),
            )
            captions = parse_caption_pair(response.text)
            if captions:
                return captions
            invalid_response_attempts += 1
            print("Format/jumlah kata belum sesuai; hasil model:")
            print(response.text or "(respons kosong)")
            if invalid_response_attempts >= 2:
                return None
        except Exception as error:
            if not is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise
            delay = min(15 * (2 ** (attempt - 1)), 60)
            print(f"Request dibatasi/gagal sementara; mencoba lagi dalam {delay} detik.")
            time.sleep(delay)


def read_completed_images():
    if not CSV_PATH.exists():
        return set()

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return {
            row["image"]
            for row in csv.DictReader(csv_file)
            if row.get("status") == "success" and row.get("image")
        }


def main():
    load_env_file()

    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Folder gambar tidak ditemukan: {IMAGE_DIR}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise SystemExit(
            f"Isi GOOGLE_API_KEY di file ini terlebih dahulu: {ENV_PATH}"
        )

    image_files = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    completed = read_completed_images()
    client = genai.Client(api_key=api_key)

    columns = [
        "image", "caption_1", "caption_2", "word_count_1",
        "word_count_2", "model", "status"
    ]
    needs_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    print(f"Model: {MODEL_ID}")
    print(f"Gambar ditemukan: {len(image_files)}; sudah selesai: {len(completed)}")
    print(f"CSV: {CSV_PATH}")

    with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if needs_header:
            writer.writeheader()

        for index, image_path in enumerate(image_files, start=1):
            if image_path.name in completed:
                print(f"[{index}/{len(image_files)}] Lewati {image_path.name}: sudah ada caption.")
                continue

            print(f"\n[{index}/{len(image_files)}] Membuat caption untuk {image_path.name}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
                captions = generate_caption(client, image_path)
                if not captions:
                    raise ValueError("Dua caption gagal dibuat atau tidak memenuhi batas kata.")
                caption_1, caption_2 = captions

                writer.writerow({
                    "image": image_path.name,
                    "caption_1": caption_1,
                    "caption_2": caption_2,
                    "word_count_1": count_words(caption_1),
                    "word_count_2": count_words(caption_2),
                    "model": MODEL_ID,
                    "status": "success",
                })
                completed.add(image_path.name)
                print(f"Caption 1 ({count_words(caption_1)} kata): {caption_1}")
                print(f"Caption 2 ({count_words(caption_2)} kata): {caption_2}")
            except Exception as error:
                error_text = str(error)
                if "API_KEY_INVALID" in error_text or "API key not valid" in error_text:
                    raise SystemExit(
                        "Google menolak GOOGLE_API_KEY. Buat/periksa key di "
                        "Google AI Studio, lalu set ulang environment variable "
                        "GOOGLE_API_KEY sebelum menjalankan skrip."
                    ) from error

                writer.writerow({
                    "image": image_path.name,
                    "caption_1": error_text,
                    "caption_2": "",
                    "word_count_1": "",
                    "word_count_2": "",
                    "model": MODEL_ID,
                    "status": "error",
                })
                print(f"Gagal memproses {image_path.name}: {error}")

            csv_file.flush()
            time.sleep(PAUSE_BETWEEN_REQUESTS)

    print("\nPembuatan caption selesai.")
    print(f"Caption sukses: {len(completed)} dari {len(image_files)} gambar.")


if __name__ == "__main__":
    main()
