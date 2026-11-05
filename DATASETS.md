# Sumber dataset tambahan

Dataset berikut cocok untuk menambah variasi gambar budaya Bali. Unduh hanya data yang diperlukan, simpan di `scraping_image/<nama_kategori>/`, dan catat sumber serta atribusi pada metadata proyek. Jangan unggah ulang gambar ke GitHub sebelum memeriksa ketentuan lisensi dan atribusi.

## Rekomendasi utama

| Dataset | Isi | Lisensi | Kegunaan |
| --- | --- | --- | --- |
| [Balinese Mask Image Dataset](https://data.mendeley.com/datasets/db79gdymts/1) | 4.710 gambar topeng Bali dalam 13 kelas; tersedia pembagian train, validation, dan test | CC BY 4.0 | Pilihan terbaik untuk menambah kategori topeng. Label kelas membantu pemeriksaan, tetapi caption tetap harus sesuai isi foto. |
| [336 Bali Traditional Textile Motif Image Dataset](https://data.mendeley.com/datasets/nc54kz78wp/1) | 56 gambar kain Bali, seperti Endek, Gringsing, dan Songket, dengan metadata jenis kain, asal, motif, teknik, dan deskripsi budaya | CC BY 4.0 | Cocok untuk kategori tekstil. Dataset menyatakan koleksinya 56 gambar meskipun angka pada judulnya 336; gunakan jumlah/file aktual setelah diunduh. |

CC BY 4.0 mengharuskan atribusi kepada pembuat, tautan lisensi, dan keterangan perubahan jika ada. Simpan informasi pencipta, URL sumber, lisensi, dan perubahan pada `metadata.csv` atau file atribusi terpisah.

## Sumber lain

- [SEA-VL melalui SEACrowd](https://seacrowd.org/projects/2025-seavl-phase-1) — dataset vision-language berskala besar berisi sekitar 1,3 juta pasangan gambar-teks dari Asia Tenggara. Cakupannya luas dan bukan khusus Bali; periksa kartu dataset untuk syarat akses dan lisensi tiap bagiannya sebelum digunakan.
- [Dataset Tari Bali di GitHub](https://github.com/widya-utami/dataset-taribali) — disebut dalam penelitian tentang klasifikasi tari Bali. Berguna untuk menambah gambar tari, tetapi cek lisensi, atribusi, dan izin redistribusi repository sebelum mengambil atau menerbitkan gambarnya.
- [Kategori Tari Bali di Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Dance_of_Bali) dan [kategori Pura di Bali](https://commons.wikimedia.org/wiki/Category:Temples_in_Bali) — sumber gambar tambahan. Lisensi berbeda untuk setiap berkas; buka halaman berkas untuk mencatat pencipta dan ketentuannya.

Skrip `scraping_image/scrap.py` mengambil gambar dari Wikimedia Commons melalui [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Search) dan mengambil URL, ukuran, serta extended metadata dari [Imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo). API Commons tersedia di [commons.wikimedia.org/w/api.php](https://commons.wikimedia.org/w/api.php). Tiap gambar tetap punya lisensi sendiri; metadata API bukan pengganti pemeriksaan halaman berkas.

## Memasukkan gambar ke proyek

1. Unduh arsip dari sumber dataset dan simpan di folder kategori, misalnya `scraping_image/topeng_bali/` atau `scraping_image/tekstil_bali/`.
2. Pertahankan label/aset sumber asli dan buat `metadata.csv` berisi minimal `image_id`, `category`, `source_url`, `license`, dan `creator`.
3. Periksa gambar rusak, duplikat, label, serta atribusi. Hindari memasukkan gambar yang lisensinya tidak diketahui ke rilis publik.
4. Jalankan `python create_annotation_template.py --image-root scraping_image --output data/annotations.csv --captions-per-image 2` untuk membuat lembar anotasi baru. Perintah ini tidak menimpa file anotasi yang sudah ada.

Folder gambar, file `.csv` metadata, dan data lokal dikecualikan oleh `.gitignore`. Simpan dataset besar di penyimpanan yang sesuai dan tautkan sumbernya di dokumentasi, alih-alih mengunggah ulang tanpa memeriksa hak distribusi.
