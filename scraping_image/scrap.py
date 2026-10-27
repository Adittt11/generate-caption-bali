
import os
import csv
import time
import hashlib
import requests
from pathlib import Path

from io import BytesIO
from PIL import Image
from ddgs import DDGS

# ==========================================
# KONFIGURASI
# ==========================================

query = "makanan tradisional bali"
folder_simpan = Path(__file__).resolve().parent / "makanan_tradisional_bali"
category = "makanan"
jumlah_gambar = 250

os.makedirs(folder_simpan, exist_ok=True)

metadata = []
hash_terdaftar = set()

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

# ==========================================
# SCRAPING GAMBAR
# ==========================================

print(f"Memulai pencarian: {query}")

try:
    with DDGS() as ddgs:
        results = ddgs.images(
            query,
            max_results=jumlah_gambar * 4
        )

        for result in results:

            if len(metadata) >= jumlah_gambar:
                break

            img_url = result.get("image")
            source_url = result.get("url")

            if not img_url:
                continue

            try:
                response = session.get(
                    img_url,
                    timeout=15
                )
                response.raise_for_status()

                # Pastikan konten merupakan gambar
                if not response.headers.get(
                    "Content-Type", ""
                ).startswith("image/"):
                    continue

                img = Image.open(
                    BytesIO(response.content)
                )
                img.verify()

                # Buka ulang untuk preprocessing
                img = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                # Filter ukuran gambar
                if img.width < 200 or img.height < 200:
                    continue

                # Cek duplikasi berdasarkan hash
                image_hash = hashlib.md5(
                    img.tobytes()
                ).hexdigest()

                if image_hash in hash_terdaftar:
                    continue

                hash_terdaftar.add(image_hash)

                # Simpan gambar
                nomor = len(metadata) + 1
                filename = f"{nomor:04d}.jpg"
                filepath = os.path.join(
                    folder_simpan,
                    filename
                )

                img.save(filepath, "JPEG", quality=95)

                # Simpan metadata
                metadata.append({
                    "image_id": filename,
                    "caption": "",
                    "category": category,
                    "search_query": query,
                    "image_url": img_url,
                    "source_url": source_url,
                    "width": img.width,
                    "height": img.height,
                    "image_hash": image_hash,
                    "license": "Perlu verifikasi"
                })

                print(
                    f"[{nomor}/{jumlah_gambar}] "
                    f"Berhasil: {filename}"
                )

                time.sleep(0.5)

            except Exception as e:
                print(f"Gagal mengunduh {img_url}: {e}")
                continue

except Exception as e:
    print("Terjadi kesalahan scraping:", e)

# ==========================================
# SIMPAN METADATA CSV
# ==========================================

csv_path = folder_simpan / "metadata.csv"

fieldnames = [
    "image_id",
    "caption",
    "category",
    "search_query",
    "image_url",
    "source_url",
    "width",
    "height",
    "image_hash",
    "license",
]

with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metadata)

print("\n================================")
print("SCRAPING SELESAI")
print("================================")
print(f"Jumlah gambar: {len(metadata)}")
print(f"Folder: {folder_simpan}")
print(f"Metadata: {csv_path}")
