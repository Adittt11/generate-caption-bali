# Pengumpulan Dataset Gambar Budaya Bali

Repository ini berisi skrip untuk mencari dan mengunduh gambar serta membuat caption untuk dataset budaya Bali. File gambar, metadata hasil scraping, dan API key disimpan lokal dan tidak dimasukkan ke GitHub.

## Persiapan

Gunakan environment Python tersendiri, lalu pasang dependensi:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Scraping gambar

Jalankan dari folder proyek:

```bash
python scraping_image/scrap.py
```

Query, kategori, jumlah maksimum gambar, dan nama folder hasil dapat diatur di bagian konfigurasi `scraping_image/scrap.py`. Skrip menyimpan gambar dan `metadata.csv` di dalam `scraping_image/`.

## Membuat caption dengan Gemini

Isi API key di `generate_caption/.env`:

```text
GOOGLE_API_KEY=PASTE_KEY_GEMINI_DI_SINI
```

Jalankan:

```bash
python generate_caption/caption.py
```

Folder gambar, nama CSV, dan prompt diatur di bagian atas `generate_caption/caption.py`. Pastikan folder gambar pada skrip caption sama dengan folder yang dihasilkan scraper. Skrip membuat dua caption per gambar dan melanjutkan gambar yang sudah berhasil saat dijalankan kembali.

## File lokal yang tidak diunggah

Aturan `.gitignore` mengecualikan gambar, file `metadata.csv`, CSV hasil caption, environment Python, serta file `.env`. Jangan menambahkan API key ke source code atau commit.
