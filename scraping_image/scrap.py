"""Download Bali culture images from Wikimedia Commons with per-file license metadata."""

import argparse
import csv
from email.utils import parsedate_to_datetime
import hashlib
import html
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = {
    "adat_bali": {
        "category": "adat",
        "keywords": ["ceremony", "ceremonial", "ritual", "tradition", "upacara", "adat", "ngaben", "melasti", "galungan", "kuningan", "mekare-kare", "pandan war"],
        "queries": [
            "Balinese traditional ceremony",
            "Balinese customary ceremony Bali",
            "Balinese Hindu ritual",
            "Ngaben Bali ceremony",
            "Melasti Bali ceremony",
            "Galungan Bali",
        ],
    },
    "busana_adat_bali": {
        "category": "busana adat",
        "keywords": ["clothing", "costume", "dress", "kebaya", "udeng", "kamen", "fashion", "wedding attire", "traditional wear", "textile in use", "pakaian", "busana", "baju adat"],
        "queries": [
            "Balinese traditional clothing",
            "Balinese traditional costume",
            "Balinese kebaya",
            "Balinese traditional dress",
            "Balinese wedding clothing",
        ],
    },
    "kerajinan_bali": {
        "category": "kerajinan",
        "keywords": ["craft", "handicraft", "carving", "woodcarving", "wood carving", "weaving", "silverwork", "silver jewelry", "silversmith", "pottery", "sculpture", "handwoven", "tenun", "songket", "endek", "gringsing", "kerajinan", "ukiran"],
        "queries": [
            "Balinese traditional craft",
            "Balinese wood carving",
            "Balinese silver jewelry",
            "Balinese weaving textile",
            "Balinese sculpture handicraft",
        ],
    },
    "makanan_tradisional_bali": {
        "category": "makanan",
        "keywords": ["food", "cuisine", "dish", "babi guling", "lawar", "sate lilit", "nasi campur", "balinese meal", "balinese cooking", "balinese kitchen", "makanan", "masakan"],
        "queries": [
            "Balinese traditional food",
            "Balinese cuisine dish",
            "Babi guling Bali",
            "Lawar Bali food",
            "Sate lilit Bali",
            "Balinese food market",
        ],
    },
    "pura_bali": {
        "category": "pura",
        "keywords": ["temple", "pura", "besakih", "uluwatu", "tanah lot", "tirta empul", "lempuyang", "batu bolong", "hindu shrine", "candi"],
        "queries": [
            "Balinese Hindu temple",
            "Pura Bali temple",
            "Besakih temple Bali",
            "Uluwatu temple Bali",
            "Tanah Lot temple Bali",
            "Tirta Empul temple Bali",
        ],
    },
    "tari_bali": {
        "category": "tari",
        "keywords": ["dance", "dancer", "kecak", "legong", "barong", "pendet", "rejang", "janger", "condong", "truna jaya", "topeng dance", "tari", "tarian", "penari"],
        "queries": [
            "Balinese traditional dance",
            "Kecak dance Bali",
            "Legong dance Bali",
            "Barong dance Bali",
            "Pendet dance Bali",
            "Rejang dance Bali",
        ],
    },
}
CSV_FIELDS = [
    "image_id", "caption", "category", "search_query", "image_url",
    "original_url", "source_url", "commons_page", "creator", "attribution",
    "license", "license_url", "commons_categories", "image_description",
    "relevance_verified",
    "width", "height", "image_hash",
]
MINIMUM_IMAGE_SIZE = 200
API_PAGE_SIZE = 50
MAX_HTTP_ATTEMPTS = 5


class RateLimitError(RuntimeError):
    """Raised when Commons continues returning 429 after respectful retries."""


def clean_metadata(value):
    """Strip HTML in Commons extended metadata and normalize whitespace."""
    if not value:
        return ""
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"</p\s*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(value).split())


def metadata_value(extmetadata, key):
    item = extmetadata.get(key) or {}
    return clean_metadata(item.get("value", ""))


def is_relevant(page, description, keywords):
    """Require the file title, description, or Commons categories to match the target."""
    category_names = [item.get("title", "") for item in page.get("categories", [])]
    evidence = " ".join([
        page.get("title", "").replace("_", " "),
        description,
        " ".join(category_names),
    ]).casefold()
    return any(keyword.casefold() in evidence for keyword in keywords), category_names


def load_existing(output_dir):
    """Read previous metadata and IDs so rerunning never overwrites images."""
    metadata_path = output_dir / "metadata.csv"
    rows = []
    if metadata_path.exists():
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append({field: row.get(field, "") for field in CSV_FIELDS})

    existing_names = {path.name for path in output_dir.glob("*.*") if path.is_file()}
    hashes = {row["image_hash"] for row in rows if row.get("image_hash")}
    numeric_ids = [
        int(match.group(1))
        for name in existing_names
        if (match := re.fullmatch(r"(\d+)\.(?:jpe?g|png|webp)", name, re.IGNORECASE))
    ]
    return rows, existing_names, hashes, max(numeric_ids, default=0), len(numeric_ids)


def count_verified_rows(rows, keywords):
    """Count only prior Commons images that passed and retain the relevance evidence."""
    count = 0
    for row in rows:
        if row.get("relevance_verified") == "yes":
            count += 1
            continue
        commons_page = row.get("commons_page", "")
        categories = row.get("commons_categories", "")
        if not commons_page or not categories:
            continue
        title = commons_page.rsplit("/", 1)[-1].replace("_", " ")
        page = {"title": title, "categories": [{"title": item} for item in categories.split(" | ")]}
        relevant, _ = is_relevant(page, row.get("image_description", ""), keywords)
        if relevant:
            count += 1
    return count


def write_metadata(output_dir, rows):
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_with_retry(session, url, params=None, timeout=30):
    """Retry transient errors and honor Wikimedia's Retry-After response header."""
    for attempt in range(MAX_HTTP_ATTEMPTS):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response

        if attempt == MAX_HTTP_ATTEMPTS - 1 and response.status_code == 429:
            raise RateLimitError(
                "Wikimedia Commons masih membatasi permintaan setelah beberapa retry. "
                "Tunggu sebelum menjalankan ulang; file dan metadata yang sudah diunduh tersimpan."
            )
        retry_after = response.headers.get("Retry-After", "")
        try:
            wait_seconds = float(retry_after)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                wait_seconds = (retry_time - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                wait_seconds = 0
        if wait_seconds <= 0:
            wait_seconds = min(5 * (2 ** attempt), 60)
        print(f"Commons membatasi request ({response.status_code}); menunggu {wait_seconds:.0f} detik sebelum retry.")
        time.sleep(wait_seconds)
    raise RuntimeError("Permintaan gagal setelah retry.")


def search_commons(session, query, result_limit):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(max(result_limit, API_PAGE_SIZE), API_PAGE_SIZE),
        "prop": "imageinfo|categories",
        "cllimit": 50,
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600,
        "iiextmetadatalanguage": "en",
        "maxlag": 5,
    }
    while True:
        response = get_with_retry(session, API_URL, params=params, timeout=30)
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"].get("info", "Wikimedia Commons API error"))
        for page in payload.get("query", {}).get("pages", []):
            yield page
        continuation = payload.get("continue")
        if not continuation:
            break
        params.update(continuation)
        time.sleep(2)


def download_category(session, folder_name, limit, delay, min_width, min_height, output_root):
    settings = CATEGORIES[folder_name]
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, existing_names, hashes, next_id, image_file_count = load_existing(output_dir)
    existing_count = count_verified_rows(rows, settings["keywords"])
    needed = max(limit - existing_count, 0)
    added = 0
    if needed == 0:
        print(f"[{folder_name}] Sudah ada {existing_count} gambar Commons relevan; target {limit} tercapai.")
        return
    if image_file_count > existing_count:
        print(f"[{folder_name}] Ada {image_file_count - existing_count} file lama yang belum lolos/verifikasi filter; file itu tetap disimpan untuk ditinjau.")
    seen_pages = {row.get("commons_page", "") for row in rows}
    skipped_relevance = 0
    skipped_license = 0
    for query in settings["queries"]:
        if added >= needed:
            break
        print(f"\n[{folder_name}] Query Commons: {query}")
        time.sleep(2)
        for page in search_commons(session, query, needed - added):
            if added >= needed:
                break
            title = page.get("title", "")
            image_info = (page.get("imageinfo") or [{}])[0]
            mime = image_info.get("mime", "")
            original_url = image_info.get("url", "")
            download_url = image_info.get("thumburl", "") or original_url
            width = int(image_info.get("width", 0) or 0)
            height = int(image_info.get("height", 0) or 0)
            extmetadata = image_info.get("extmetadata", {})
            description = metadata_value(extmetadata, "ImageDescription")
            relevant, category_names = is_relevant(page, description, settings["keywords"])
            if not relevant:
                skipped_relevance += 1
                continue
            license_name = metadata_value(extmetadata, "LicenseShortName")
            license_url = metadata_value(extmetadata, "LicenseUrl")
            if not title or not mime.startswith("image/") or not download_url:
                continue
            if width < min_width or height < min_height:
                continue
            # Require Commons to provide an explicit license before downloading.
            if not license_name:
                skipped_license += 1
                continue

            commons_page = "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":()')!,")
            if commons_page in seen_pages:
                continue

            try:
                response = get_with_retry(session, download_url, timeout=30)
                with Image.open(BytesIO(response.content)) as opened:
                    image = opened.convert("RGB")
                if image.width < min_width or image.height < min_height:
                    continue
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                if image_hash in hashes:
                    seen_pages.add(commons_page)
                    continue

                next_id += 1
                filename = f"{next_id:04d}.jpg"
                while filename in existing_names:
                    next_id += 1
                    filename = f"{next_id:04d}.jpg"
                image.save(output_dir / filename, "JPEG", quality=95)

                creator = metadata_value(extmetadata, "Artist")
                attribution = metadata_value(extmetadata, "Attribution") or creator
                rows.append({
                    "image_id": filename,
                    "caption": "",
                    "category": settings["category"],
                    "search_query": query,
                    "image_url": download_url,
                    "original_url": original_url,
                    "source_url": commons_page,
                    "commons_page": commons_page,
                    "creator": creator,
                    "attribution": attribution,
                    "license": license_name,
                    "license_url": license_url,
                    "commons_categories": " | ".join(category_names),
                    "image_description": description,
                    "relevance_verified": "yes",
                    "width": image.width,
                    "height": image.height,
                    "image_hash": image_hash,
                })
                hashes.add(image_hash)
                existing_names.add(filename)
                seen_pages.add(commons_page)
                added += 1
                print(f"[{existing_count + added}/{limit}] {filename}: {title} ({license_name})")
                write_metadata(output_dir, rows)
                if description:
                    print(f"  Deskripsi Commons: {description[:180]}")
                time.sleep(delay)
            except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as error:
                print(f"Gagal mengunduh {title}: {error}")

    write_metadata(output_dir, rows)
    final_count = existing_count + added
    print(f"Selesai {folder_name}: {final_count}/{limit} gambar; ditambahkan {added} gambar baru.")
    print(f"Hasil dilewati: {skipped_relevance} tidak cocok kategori, {skipped_license} tanpa metadata lisensi.")


def main():
    parser = argparse.ArgumentParser(
        description="Unduh gambar budaya Bali dari Wikimedia Commons beserta metadata lisensinya."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--category", choices=sorted(CATEGORIES))
    selection.add_argument("--all", action="store_true", help="Scrape semua enam kategori")
    parser.add_argument("--limit", type=int, default=250, help="Target gambar Commons relevan per kategori")
    parser.add_argument("--delay", type=float, default=0.5, help="Jeda antar-unduhan dalam detik")
    parser.add_argument("--min-width", type=int, default=MINIMUM_IMAGE_SIZE)
    parser.add_argument("--min-height", type=int, default=MINIMUM_IMAGE_SIZE)
    parser.add_argument("--output-root", default="dataset/images", help="Folder induk dataset gambar")
    args = parser.parse_args()
    if args.limit < 1 or args.delay < 0:
        parser.error("--limit harus positif dan --delay tidak boleh negatif")

    user_agent = os.getenv("WIKIMEDIA_USER_AGENT", "Dataset100BaliCaption/1.0 (educational image research)")
    if "contact" not in user_agent.casefold() and "@" not in user_agent:
        print("Disarankan mengatur WIKIMEDIA_USER_AGENT dengan email atau URL kontak untuk mengurangi blokir API.")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    # Preserve the old no-argument workflow: scrape all six categories.
    if args.category:
        categories = {args.category: CATEGORIES[args.category]}
    else:
        categories = CATEGORIES
    for folder_name in categories:
        try:
            download_category(
                session, folder_name, args.limit, args.delay,
                args.min_width, args.min_height, output_root,
            )
        except RateLimitError as error:
            print(f"Scraping dihentikan agar tidak terus membebani Commons: {error}")
            break
        except (requests.RequestException, RuntimeError) as error:
            print(f"Kategori {folder_name} gagal diproses: {error}")


if __name__ == "__main__":
    main()
