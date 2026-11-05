# Prompt verifikasi kategori dan caption budaya Bali

Pakai satu gambar per permintaan pada model yang dapat melihat gambar. Sertakan nama kategori folder dan metadata sumber bila tersedia.

```text
Periksa gambar ini untuk dataset caption budaya Bali.

Kategori folder yang diharapkan: [ISI SATU: adat_bali | busana_adat_bali | kerajinan_bali | makanan_tradisional_bali | pura_bali | tari_bali]
Metadata sumber yang sudah tersedia (jika ada): [JUDUL/SUMBER/LOKASI, ATAU "tidak tersedia"]

Tentukan dua hal secara terpisah:
1. Apakah gambar memiliki bukti yang cukup bahwa subjeknya terkait Bali?
2. Apakah subjek utama gambar benar-benar sesuai dengan kategori folder yang diharapkan?

Gunakan hanya bukti visual dan metadata sumber yang diberikan. Jangan menyimpulkan asal Bali hanya dari orang berwajah Asia, pakaian umum, arsitektur yang tampak mirip, atau latar tropis. Jangan menganggap gambar sebagai budaya Bali hanya karena nama file atau kata pencarian menyebut Bali. Jika bukti kurang jelas, pilih "review", bukan menebak.

Aturan keputusan:
- "keep": bukti Bali cukup DAN subjek utama cocok dengan kategori.
- "reject": gambar jelas bukan tentang Bali ATAU jelas masuk kategori lain.
- "review": asal atau kategorinya tidak dapat dipastikan dari bukti yang tersedia.
- Beri confidence "tinggi", "sedang", atau "rendah" sesuai kekuatan bukti.

Jika keputusan "keep", tulis DUA caption berbeda dalam Bahasa Indonesia. Setiap caption harus tepat satu kalimat dan terdiri dari 10 sampai 15 kata, dihitung berdasarkan kata yang dipisahkan spasi. Deskripsikan hanya subjek, atribut, tindakan, dan latar yang terlihat. Hindari klaim lokasi spesifik, nama ritual, makna simbol, bahan, atau sejarah kecuali didukung metadata tepercaya. Kedua caption harus akurat dan tidak mengulang kalimat yang sama.

Jika keputusan "reject" atau "review", isi kedua caption dengan string kosong.

Keluarkan JSON valid saja, tanpa markdown atau teks tambahan:
{
  "decision": "review",
  "category_match": true,
  "bali_evidence": "bukti visual atau metadata yang mendukung; jika tidak ada, tulis tidak cukup",
  "reason": "alasan singkat",
  "confidence": "tinggi | sedang | rendah",
  "caption_1": "",
  "caption_2": ""
}
```

## Contoh caption untuk kategori tari Bali

```json
{
  "decision": "keep",
  "category_match": true,
  "bali_evidence": "Metadata sumber menyebut pertunjukan tari Bali.",
  "reason": "Penari dan pertunjukan terlihat jelas.",
  "confidence": "tinggi",
  "caption_1": "Seorang penari Bali mengenakan busana tradisional berwarna cerah sambil memegang kipas di panggung terbuka.",
  "caption_2": "Para penari tampil berbaris dengan kain bermotif dan hiasan kepala saat pertunjukan berlangsung."
}
```

Contoh caption masing-masing berisi 14 dan 13 kata. Tetap periksa gambar sendiri sebelum memberi keputusan `keep` atau mengubah `reviewed` menjadi `yes`.
