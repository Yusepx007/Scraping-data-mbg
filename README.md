# Scraping Data MBG

Script web scraping untuk mengumpulkan teks berita online mengenai Program Makan Bergizi Gratis (MBG) dari berbagai portal berita.

## Fitur & Alur Kerja
1. Mengambil berita dari feed RSS Google News atau daftar URL portal berita.
2. Mengekstrak judul, tanggal, sumber, dan konten artikel berita.
3. Melakukan tokenisasi kalimat dengan NLTK.
4. Menyimpan dataset ke dalam format CSV (`dataset_mbg_scraping.csv` / `dataset_mbg.csv`).

## Cara Penggunaan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Scraping
```bash
python scraping_berita_mbg.py
```
