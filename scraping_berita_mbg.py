"""
scraping_berita_mbg.py
=======================
Script web scraping untuk mengumpulkan teks berita online mengenai
Program Makan Bergizi Gratis (MBG) dari berbagai portal berita.

ALUR KERJA:
1. Membaca daftar URL artikel berita dari variabel DAFTAR_URL (bisa juga
   dipindah ke file urls.txt, satu URL per baris).
2. Mengambil (fetch) HTML setiap halaman menggunakan requests.
3. Mengekstrak sumber, tanggal terbit, dan isi artikel menggunakan
   BeautifulSoup dengan beberapa pola selector umum portal berita
   Indonesia (fallback ke seluruh tag <p> bila pola tidak ditemukan).
4. Membersihkan teks lalu memecahnya menjadi kalimat (sentence
   tokenization) menggunakan NLTK.
5. Menyimpan setiap kalimat sebagai satu baris data ke file CSV dengan
   kolom: no, teks_asli, sumber, tanggal, url, sentimen (kolom sentimen
   dikosongkan untuk diisi secara manual pada tahap pelabelan / BAB IV).

Cara pakai:
    1. Isi DAFTAR_URL dengan tautan artikel berita yang ingin dikumpulkan.
    2. Jalankan: python scraping_berita_mbg.py
    3. Hasil tersimpan di dataset_mbg_scraping.csv pada folder yang sama.

Dependensi:
    pip install requests beautifulsoup4 pandas nltk
"""

import re
import time
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}

JEDA_ANTAR_REQUEST = 2          # detik, agar tidak membebani server sumber
MIN_KATA_PER_KALIMAT = 5        # abaikan kalimat yang terlalu pendek/tidak bermakna
TARGET_BARIS = 400              # target jumlah baris data yang ingin dikumpulkan
MAX_ARTIKEL_RSS = 30            # jumlah artikel berita yang diambil dari RSS
RSS_QUERIES = [
    "Program Makan Bergizi Gratis",
    "Makan Bergizi Gratis",
    "Program MBG",
    "MBG sekolah",
]
OUTPUT_CSV = "dataset_mbg_scraping.csv"


# ---------------------------------------------------------------
# 1. DAFTAR URL ARTIKEL YANG AKAN DI-SCRAPE
#    Tambahkan / ganti dengan URL artikel berita mengenai Program MBG
#    dari portal-portal berikut (atau portal lain).
# ---------------------------------------------------------------
DAFTAR_URL = [
    "https://www.antaranews.com/berita/xxxxxx/judul-artikel-mbg",
    "https://www.kompas.com/xxxxxx/judul-artikel-mbg",
    "https://news.detik.com/berita/xxxxxx/judul-artikel-mbg",
    "https://www.republika.co.id/berita/xxxxxx/judul-artikel-mbg",
    # tambahkan URL lain sesuai kebutuhan penelitian ...
]


# Pemetaan domain -> nama sumber yang rapi (ditampilkan pada kolom 'sumber')
PEMETAAN_SUMBER = {
    "antaranews.com": "Antara News",
    "kompas.com": "Kompas.com",
    "detik.com": "Detik.com",
    "republika.co.id": "Republika.co.id",
    "wikipedia.org": "Wikipedia",
    "kemenkeu.go.id": "Kemenkeu RI",
    "setneg.go.id": "Sekretariat Negara RI",
    "bgn.go.id": "Badan Gizi Nasional",
    "bpmpsumut.kemdikbud.go.id": "BPMP Sumatera Utara",
}

# Selector CSS umum untuk badan artikel pada beberapa portal populer.
# Ditambahkan berurutan dari yang paling spesifik ke paling umum.
SELECTOR_KONTEN = [
    "div.detail__body-text",   # detik.com
    "div.read__content",       # kompas.com
    "div.post-content",        # antaranews.com
    "div#artikel",             # republika.co.id (contoh)
    "div.article-content",
    "article",
]


def deteksi_sumber(url: str) -> str:
    """Menentukan nama sumber berita berdasarkan domain URL."""
    domain = urlparse(url).netloc.replace("www.", "")
    for kunci, nama in PEMETAAN_SUMBER.items():
        if kunci in domain:
            return nama
    return domain


def ambil_tanggal(soup: BeautifulSoup) -> str:
    """Mengekstrak tanggal publikasi dari beberapa pola meta tag umum."""
    pola_meta = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"itemprop": "datePublished"}),
        ("meta", {"name": "date"}),
    ]
    for tag, attrs in pola_meta:
        el = soup.find(tag, attrs=attrs)
        if el and el.get("content"):
            return el["content"][:10]

    tag_time = soup.find("time")
    if tag_time and tag_time.get("datetime"):
        return tag_time["datetime"][:10]

    return ""


def ambil_konten_artikel(soup: BeautifulSoup) -> str:
    """
    Mengekstrak isi artikel dengan mencoba beberapa selector umum yang
    dipakai portal berita Indonesia. Jika tidak ada yang cocok, fallback
    ke seluruh tag <p> pada halaman.
    """
    for selector in SELECTOR_KONTEN:
        container = soup.select_one(selector)
        if container:
            paragraf = container.find_all("p")
            if paragraf:
                return " ".join(p.get_text(strip=True) for p in paragraf)

    paragraf = soup.find_all("p")
    return " ".join(p.get_text(strip=True) for p in paragraf)


def bersihkan_teks(teks: str) -> str:
    """Merapikan whitespace dan membuang boilerplate umum (baca juga, dsb)."""
    teks = re.sub(r"\s+", " ", teks)
    teks = re.sub(
        r"(Baca juga|Baca Juga|ADVERTISEMENT|Scroll to continue with content).*?(?=[A-Z])",
        "",
        teks,
    )
    return teks.strip()


def scrape_artikel(url: str) -> dict | None:
    """Mengambil dan mengekstrak satu artikel dari URL yang diberikan."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Gagal mengambil {url} -> {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    sumber = deteksi_sumber(url)
    tanggal = ambil_tanggal(soup)
    konten = bersihkan_teks(ambil_konten_artikel(soup))

    return {"sumber": sumber, "tanggal": tanggal, "url": url, "konten": konten}


def ambil_item_rss(query: str, max_artikel: int = 30) -> list[dict]:
    """Mengambil item berita dari feed RSS Google News."""
    rss_url = (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}&hl=id&gl=ID&ceid=ID:id"
    )
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Gagal mengambil RSS: {e}")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        print(f"  Gagal parsing RSS: {e}")
        return []

    items = []
    for item in root.findall("./channel/item")[:max_artikel]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        desc_html = item.findtext("description") or ""
        desc = ""
        if desc_html:
            desc = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)

        konten = " ".join([bagian for bagian in [title, desc] if bagian]).strip()
        if konten:
            items.append(
                {
                    "title": title,
                    "konten": konten,
                    "url": link,
                    "tanggal": pub_date,
                }
            )
    return items


def main():
    hasil = []
    seen_urls = set()

    daftar_url = [
        url for url in DAFTAR_URL
        if url and "xxxxxx" not in url and "judul-artikel-mbg" not in url
    ]

    if daftar_url:
        print("Menggunakan daftar URL yang sudah ada...")
        sumber_items = [(url, None) for url in daftar_url]
    else:
        print("Mengambil data dari RSS Google News...")
        sumber_items = []
        for query in RSS_QUERIES:
            for item in ambil_item_rss(query, MAX_ARTIKEL_RSS):
                sumber_items.append((item["url"], item))

    print(f"Total item yang akan diproses: {len(sumber_items)}")

    for i, (url, item) in enumerate(sumber_items, 1):
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        if item is None:
            print(f"[{i}/{len(sumber_items)}] Scraping: {url}")
            data = scrape_artikel(url)
            if data and data["konten"]:
                konten = data["konten"]
                sumber = data["sumber"]
                tanggal = data["tanggal"]
            else:
                continue
        else:
            print(f"[{i}/{len(sumber_items)}] Menggunakan item RSS: {item['title']}")
            konten = item["konten"]
            sumber = "Google News"
            tanggal = item["tanggal"]
            url = item["url"]

        kalimat_list = sent_tokenize(konten)
        for kalimat in kalimat_list:
            kalimat = kalimat.strip()
            if len(kalimat.split()) >= MIN_KATA_PER_KALIMAT:
                hasil.append(
                    {
                        "teks_asli": kalimat,
                        "sumber": sumber,
                        "tanggal": tanggal,
                        "url": url,
                        "sentimen": "",  # diisi manual pada tahap pelabelan
                    }
                )
                if len(hasil) >= TARGET_BARIS:
                    break
        if len(hasil) >= TARGET_BARIS:
            break

        time.sleep(JEDA_ANTAR_REQUEST)

    if not hasil:
        print("\nTidak ada data yang berhasil dikumpulkan. Periksa URL sumber.")
        return

    df = pd.DataFrame(hasil)
    df.insert(0, "no", range(1, len(df) + 1))
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nScraping selesai. Total {len(df)} baris data disimpan ke '{OUTPUT_CSV}'.")
    print("Kolom 'sentimen' masih kosong -> lanjutkan ke tahap pelabelan manual")
    print("sebelum data digunakan pada proses pra-pemrosesan dan klasifikasi.")


if __name__ == "__main__":
    main()
