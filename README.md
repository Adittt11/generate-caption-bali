# Image Captioning Budaya Bali

Proyek ini membuat caption Bahasa Indonesia untuk gambar budaya Bali. Model utama adalah BLIP `Salesforce/blip-image-captioning-base` yang di-fine-tune dengan caption yang sudah ditinjau manusia. Inferensi berjalan lokal melalui Python atau FastAPI; tidak ada layanan Gemini/OpenAI API di jalur prediksi.

Model awal tersedia dengan lisensi BSD-3-Clause di [Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base). Training dan evaluasi mengikuti pola image-caption pair dari [panduan image captioning Transformers](https://huggingface.co/docs/transformers/tasks/image_captioning).

## Struktur

- `scraping_image/scrap.py` — mencari dan mengunduh gambar serta metadata.
- `create_annotation_template.py` — membuat lembar anotasi untuk gambar lokal.
- `prepare_dataset.py` — memvalidasi label dan membuat split train/validation/test per gambar.
- `train_captioner.py` — fine-tuning BLIP dan menyimpan checkpoint terbaik berdasarkan validation loss.
- `evaluate_captioner.py` — membuat prediksi test dan menghitung BLEU-4 serta ROUGE-L.
- `generate_caption/caption.py` — inferensi lokal lewat command line.
- `api/app.py` — API FastAPI lokal untuk inference.

## 1. Persiapan environment

Gunakan virtual environment agar paket proyek tidak bentrok dengan environment Python lain.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Untuk Google Colab, pasang dependensi dari `requirements.txt` di notebook. Pilih GPU runtime untuk training yang lebih cepat.

## 2. Mengumpulkan gambar

Atur `query`, `folder_simpan`, `category`, dan `jumlah_gambar` di bagian konfigurasi `scraping_image/scrap.py`, lalu jalankan dari root proyek:

```bash
python scraping_image/scrap.py
```

Gambar dan `metadata.csv` disimpan di folder dalam `scraping_image/`. Periksa duplikasi, kualitas, sumber, dan izin penggunaan gambar sebelum anotasi.

## 3. Membuat dan meninjau caption

Buat template dengan dua caption referensi untuk tiap gambar:

```bash
python create_annotation_template.py --image-root scraping_image --output data/annotations.csv --captions-per-image 2
```

Isi kolom `caption` secara manual. Set `reviewed` menjadi `yes` setelah caption diperiksa manusia. Caption harus mendeskripsikan hal yang terlihat dan tidak mengarang nama lokasi, sejarah, atau makna simbol. Setiap baris adalah satu referensi caption; dua baris dengan `image_path` yang sama adalah dua referensi untuk gambar yang sama.

Label yang belum diisi atau belum ditinjau tidak akan digunakan. Gemini/OpenAI dan layanan sejenis tidak dipakai untuk menghasilkan prediksi model utama.

## 4. Membagi dataset

```bash
python prepare_dataset.py --annotations data/annotations.csv --output-dir data/splits --seed 42
```

Split dibuat per `image_path`, sehingga caption referensi untuk satu gambar tidak tersebar ke train, validation, dan test. Untuk menjaga evaluasi bermakna, kumpulkan cukup gambar di setiap kategori agar semua split terisi.

## 5. Fine-tuning

```bash
python train_captioner.py --train-csv data/splits/train.csv --validation-csv data/splits/validation.csv --epochs 5 --batch-size 2
```

Checkpoint dengan validation loss terbaik disimpan ke `outputs/blip-finetuned/`. Opsi `--device auto` memilih CUDA, Apple MPS, atau CPU. Training paling sesuai dijalankan di Google Colab GPU bila perangkat lokal lambat atau kekurangan memori.

## 6. Evaluasi dan baseline

Jalankan evaluasi pada test split yang tidak dipakai untuk training atau pemilihan checkpoint.

Baseline pre-trained sebelum fine-tuning:

```bash
python evaluate_captioner.py --test-csv data/splits/test.csv --model Salesforce/blip-image-captioning-base --output-dir outputs/baseline
```

Model fine-tuned:

```bash
python evaluate_captioner.py --test-csv data/splits/test.csv --model outputs/blip-finetuned --output-dir outputs/fine-tuned
```

Hasil metrik tersimpan di `metrics.json`, dan contoh prediksi beserta referensi tersimpan di `predictions.csv`. BLEU-4 mengukur kesamaan n-gram hingga empat kata; ROUGE-L mengukur kesamaan urutan terpanjang. Laporkan kedua skor dan tinjau contoh prediksi secara manual karena skor otomatis tidak sepenuhnya mengukur ketepatan makna budaya.

## 7. Inferensi lokal dan API

Caption satu gambar dari terminal:

```bash
python generate_caption/caption.py scraping_image/kerajinan_bali/0001.jpg --checkpoint outputs/blip-finetuned
```

Jalankan API:

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Dokumentasi interaktif tersedia di `http://127.0.0.1:8000/docs`. Endpoint `POST /caption` menerima file gambar pada field form `image` dan mengembalikan caption. Model dimuat dari checkpoint fine-tuned lokal; API akan menolak berjalan jika checkpoint belum tersedia.

## 8. GitHub dan laporan

`data/`, `outputs/`, gambar, metadata hasil, dan file `.env` dikecualikan oleh `.gitignore`. Repository berisi source code dan instruksi; simpan dataset hanya jika lisensinya mengizinkan. Jika checkpoint terlalu besar untuk GitHub, unggah ke Google Drive dan tambahkan tautan unduh ke README.

Untuk laporan, bandingkan baseline dengan model fine-tuned, jelaskan BLEU-4 dan ROUGE-L, analisis contoh prediksi berhasil/gagal, dan cantumkan minimal lima artikel jurnal Scopus terbit dalam lima tahun terakhir setelah status indeks dan tahunnya diverifikasi.
