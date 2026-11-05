# Image Captioning Budaya Bali

Proyek ini menghasilkan caption Bahasa Indonesia untuk foto budaya Bali. Model usulan memakai CNN pre-trained sebagai encoder visual dan Transformer autoregresif sebagai decoder teks. Tidak menggunakan BLIP.

## Rancangan model

```text
Foto → CNN encoder (ResNet-50 / EfficientNet-B0) → grid fitur + positional encoding
     → Transformer decoder (masked self-attention + cross-attention)
     → caption Bahasa Indonesia (beam search)
```

Encoder CNN mengekstrak grid fitur visual. Decoder menghasilkan teks kata demi kata: masked self-attention hanya melihat token sebelumnya, sementara cross-attention menghubungkan token dengan fitur gambar. Decoder Transformer dapat menghitung posisi teks secara paralel saat training. Saat inferensi, caption tetap dibuat autoregresif dengan beam search.

## Struktur

```text
dataset100/
├── api/                       # API FastAPI untuk model terlatih
├── data/                      # Dataset lokal, anotasi, review, dan split (tidak masuk Git)
├── dataset/
│   └── images/                # Gambar lokal per kategori (tidak masuk Git)
├── generate_caption/          # Generator caption Gemini dan inferensi CNN lokal
├── models/                    # CNN encoder, Transformer decoder, baseline LSTM
├── prompts/                   # Prompt pemeriksaan budaya dan caption
├── scraping_image/            # Kode scraper Wikimedia Commons
├── scripts/
│   └── organize_dataset.py     # Memindahkan dan menomori gambar per kategori
├── create_annotation_template.py
├── review_dataset.py
├── prepare_dataset.py
├── train_captioner.py
├── evaluate_captioner.py
├── requirements.txt
├── DATASETS.md
└── README.md
```

`data/`, `dataset/images/`, cache, dan checkpoint adalah artefak lokal yang dikecualikan dari Git. File sumber kode, dokumentasi, dan `.env.example` yang tidak berisi API key dapat dimasukkan ke repository.

## 1. Persiapan environment

Gunakan virtual environment. Google Colab dengan GPU disarankan untuk training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Dataset dan anotasi

Skrip `scraping_image/scrap.py` menggunakan Wikimedia Commons Action API, bukan DDGS. Commons memberi halaman sumber dan metadata lisensi/pencipta untuk tiap gambar. Hasil yang lisensinya tidak dicantumkan atau tidak cocok dengan kategori dilewati. Target default 250 gambar Commons yang lolos filter relevansi per kategori.

Rapikan dataset ke folder `dataset/images/` dan urutkan nama file per kategori. Perintah pertama hanya menampilkan rencana; tambahkan `--apply` untuk memindahkan dan mengganti nama file. Skrip juga menyesuaikan metadata, CSV anotasi/review, CSV caption, contact sheet, dan menyimpan peta nama lama-baru di `data/filename_rename_manifest.csv`.

```bash
python scripts/organize_dataset.py
python scripts/organize_dataset.py --apply
```

Scrape satu kategori. Hasil disimpan ke `dataset/images/<kategori>/`:

```bash
python scraping_image/scrap.py --category tari_bali --limit 250
```

Kategori yang tersedia: `adat_bali`, `busana_adat_bali`, `kerajinan_bali`, `makanan_tradisional_bali`, `pura_bali`, dan `tari_bali`. Tanpa argumen, skrip otomatis menjalankan semua enam kategori agar tetap mudah dipakai dengan `python scrap.py` (target hingga 1.500 gambar Commons relevan). Skrip menambahkan kekurangan saja saat dijalankan ulang. File lama yang belum terverifikasi tetap disimpan dan tidak dihitung ke target; karena itu jumlah file di folder dapat lebih dari 250. Untuk menjalankan semuanya secara eksplisit:

```bash
python scraping_image/scrap.py --all --limit 250
```

Gambar ditambahkan ke folder kategori yang sesuai. Skrip mencari melalui beberapa frasa dan halaman hasil, lalu hanya mengunduh jika judul, deskripsi, atau kategori berkas Commons cocok dengan kata kunci kategori. Skrip juga mempertahankan gambar dan metadata yang sudah ada, mencegah duplikasi berdasarkan hash, serta mencatat pencipta, lisensi, kategori Commons, dan halaman sumber setelah tiap unduhan. Filter teks membantu mengurangi hasil salah kategori, tetapi gambar tetap perlu ditinjau manual. Gambar lama hasil DDGS tidak dihapus otomatis dan tidak dihitung ke target 250 gambar Commons terverifikasi; folder dapat berisi lebih dari 250 file sampai gambar lama ditinjau. Periksa halaman sumber dan syarat atribusi sebelum penggunaan. File gambar dan data lokal tidak masuk Git secara default.

Atur User-Agent dengan kontak pengelola sebelum menjalankan scrape agar Wikimedia dapat menghubungi pemilik aplikasi jika permintaan dibatasi:

```bash
export WIKIMEDIA_USER_AGENT="Dataset100BaliCaption/1.0 (email: GANTI_DENGAN_EMAIL_ANDA)"
```

Saat menerima HTTP 429, skrip mengikuti `Retry-After` atau memakai exponential backoff. Jika pembatasan terus terjadi, skrip menghentikan proses; jalankan kembali setelah beberapa waktu. Permintaan dilakukan berurutan dengan jeda.

Sebelum anotasi, audit gambar yang sudah terkumpul. Skrip membuat contact sheet per folder serta CSV untuk mencatat keputusan. Buka contact sheet, isi `decision` di CSV dengan `keep`, `reject`, atau `review`, lalu koreksi `actual_category` bila gambar masuk kategori lain:

```bash
python review_dataset.py --image-root dataset/images --output-dir data/review
```

Gunakan prompt pada `prompts/caption_prompt_id.md` untuk mengecek satu gambar atau membuat caption. Prompt meminta bukti bahwa gambar terkait Bali dan cocok dengan kategori, serta memberi status `keep`, `reject`, atau `review`.

Buat lembar anotasi hanya dari gambar yang ditandai `keep`:

```bash
python create_annotation_template.py --image-root dataset/images --review-csv data/review/review.csv --output data/annotations_curated.csv --captions-per-image 2
```

Set API key Gemini sebagai environment variable. Jangan tulis API key langsung di kode atau commit ke GitHub:

```bash
export GEMINI_API_KEY="API_KEY_DARI_GOOGLE_AI_STUDIO"
```

Caption awal dapat dibuat langsung ke CSV. Model mengembalikan teks biasa; skrip memeriksa dua caption per gambar dan batas 10–15 kata. Gambar yang berhasil diproses disimpan satu per satu, jadi perintah dapat dijalankan ulang untuk melanjutkan:

```bash
python generate_caption/caption.py --annotations data/annotations_curated.csv
```

Setiap caption juga diminta menyebut "Bali" atau unsur budaya Bali yang sesuai kategori gambar. Untuk membuat ulang semua caption dengan aturan itu, gunakan `--overwrite`; CSV sebelumnya dicadangkan di `data/backups/`. Jika proses terhenti karena kuota, jalankan kembali tanpa `--overwrite` agar gambar yang sudah selesai tidak diproses ulang.

```bash
python generate_caption/caption.py --annotations data/annotations_curated.csv --overwrite
```

Untuk mencoba satu gambar lebih dulu, gunakan `--limit 1`. Model default adalah `gemini-3.5-flash-lite`; ubah dengan `--gemini-model` bila diperlukan. Caption otomatis adalah draft: cocokkan dengan gambar, lalu ubah kolom `reviewed` menjadi `yes` setelah disetujui. Jangan mengarang lokasi, nama upacara, sejarah, atau makna simbol.

Buat split train/validation/test berdasarkan gambar agar gambar yang sama tidak bocor ke beberapa split:

```bash
python prepare_dataset.py --annotations data/annotations_curated.csv --output-dir data/splits --seed 42
```

Split dibuat per `image_path`, agar gambar yang sama tidak tersebar ke train, validation, dan test. Baris kosong/belum ditinjau serta caption di luar batas 10–15 kata akan dilewati.

## 3. Model usulan: CNN–Transformer

Training berjalan dalam dua tahap. Tahap pertama membekukan seluruh CNN dan melatih decoder. Tahap kedua membuka blok akhir CNN dan melatihnya dengan learning rate lebih kecil daripada decoder. Checkpoint terbaik tiap tahap dipilih dari validation loss; model dengan loss validasi lebih rendah disimpan sebagai `model.pt`.

```bash
python train_captioner.py \
  --train-csv data/splits/train.csv \
  --validation-csv data/splits/validation.csv \
  --backbone resnet50 \
  --decoder transformer \
  --stage1-epochs 5 \
  --stage2-epochs 10 \
  --batch-size 16 \
  --decoder-lr 0.0003 \
  --encoder-lr 0.00001 \
  --warmup-steps 100 \
  --patience 3 \
  --max-words 15 \
  --output-dir outputs/cnn-transformer
```

Backbone alternatif: `--backbone efficientnet_b0`. Augmentasi horizontal dan warna dapat diaktifkan dengan `--augment`. Untuk GPU Colab gunakan `--device cuda`; default `auto` memilih CUDA, Apple MPS, atau CPU.

## 4. Baseline CNN–LSTM dan eksperimen

Baseline memakai backbone dan split yang sama, tetapi decoder LSTM tanpa attention. Jalankan sebagai eksperimen terpisah:

```bash
python train_captioner.py \
  --train-csv data/splits/train.csv \
  --validation-csv data/splits/validation.csv \
  --backbone resnet50 \
  --decoder lstm \
  --output-dir outputs/cnn-lstm-resnet50
```

Evaluasi CNN–Transformer dan CNN–LSTM pada test set yang sama. Untuk perbandingan yang adil, samakan data, seed, batch size, augmentasi, dan aturan early stopping. Variasi eksperimen yang dapat dilaporkan:

| Variabel | Kondisi |
| --- | --- |
| Backbone CNN | ResNet-50, EfficientNet-B0 |
| Decoder | Transformer, LSTM tanpa attention |
| Fine-tuning CNN | Dibekukan, blok terakhir dibuka |
| Augmentasi | Mati, hidup |
| Decoder Transformer | Jumlah layer dan attention head |

Jalankan setiap konfigurasi pada output directory berbeda. Catat BLEU-4, ROUGE-L, loss validasi terbaik, waktu training, dan contoh caption yang perlu diperbaiki.

## 5. Evaluasi

```bash
python evaluate_captioner.py \
  --test-csv data/splits/test.csv \
  --model outputs/cnn-transformer \
  --output-dir outputs/evaluation/cnn-transformer

python evaluate_captioner.py \
  --test-csv data/splits/test.csv \
  --model outputs/cnn-lstm-resnet50 \
  --output-dir outputs/evaluation/cnn-lstm-resnet50
```

Hasil disimpan dalam `metrics.json` dan `predictions.csv`. BLEU-4 dan ROUGE-L membandingkan kecocokan teks, tetapi tidak cukup untuk mengukur ketepatan budaya; tinjau juga prediksi secara manual.

## 6. Inferensi lokal dan API

Caption satu gambar:

```bash
python generate_caption/caption.py dataset/images/tari_bali/0001.jpg \
  --checkpoint outputs/cnn-transformer --beam-size 3
```

Jalankan API:

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Dokumentasi interaktif tersedia di `http://127.0.0.1:8000/docs`. Endpoint `POST /caption` menerima file gambar pada field form `image`. API memuat checkpoint lokal dari `outputs/cnn-transformer`; ubah `CAPTION_MODEL_DIR` untuk memilih checkpoint lain.

## 7. GitHub dan atribusi

`data/`, `outputs/`, gambar, metadata hasil, dan file `.env` dikecualikan oleh `.gitignore`. Lihat `DATASETS.md` untuk sumber tambahan serta ketentuan atribusi sebelum memakai atau membagikan dataset. Jangan commit checkpoint besar ke GitHub; bagikan melalui penyimpanan yang sesuai dan dokumentasikan cara mendapatkannya.
