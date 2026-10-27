import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = ["image_path", "caption", "category", "reviewed"]
SPLITS = ("train", "validation", "test")


def load_approved_rows(csv_path):
    rows = []
    skipped = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Kolom wajib hilang di CSV: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            caption = (row.get("caption") or "").strip()
            reviewed = (row.get("reviewed") or "").strip().lower()
            if not image_path or not caption or reviewed not in {"yes", "ya", "true", "1"}:
                skipped += 1
                continue

            resolved_image = Path(image_path)
            if not resolved_image.is_absolute():
                resolved_image = ROOT / resolved_image
            if not resolved_image.is_file():
                raise FileNotFoundError(
                    f"Gambar pada baris {line_number} tidak ditemukan: {resolved_image}"
                )

            rows.append({
                "image_path": Path(image_path).as_posix(),
                "caption": caption,
                "category": (row.get("category") or "").strip(),
                "reviewed": "yes",
            })
    return rows, skipped


def assign_image_splits(rows, seed):
    rng = random.Random(seed)
    image_paths = sorted({row["image_path"] for row in rows})
    rng.shuffle(image_paths)
    count = len(image_paths)
    train_count = max(1, int(count * 0.8))
    validation_count = max(1, int(count * 0.1))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1

    assignment = {}
    first, second = train_count, train_count + validation_count
    for image_path in image_paths[:first]:
        assignment[image_path] = "train"
    for image_path in image_paths[first:second]:
        assignment[image_path] = "validation"
    for image_path in image_paths[second:]:
        assignment[image_path] = "test"
    return assignment


def main():
    parser = argparse.ArgumentParser(
        description="Validasi anotasi dan pisahkan data per gambar agar tidak bocor antar-split."
    )
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations_path = (ROOT / args.annotations).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if not annotations_path.is_file():
        raise SystemExit(f"CSV anotasi tidak ditemukan: {annotations_path}")

    rows, skipped = load_approved_rows(annotations_path)
    image_count = len({row["image_path"] for row in rows})
    if image_count < 3:
        raise SystemExit("Perlu minimal 3 gambar dengan caption yang sudah ditinjau.")

    assignments = assign_image_splits(rows, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {name: [] for name in SPLITS}
    for row in rows:
        split_rows[assignments[row["image_path"]]].append(row)

    for split_name, split_data in split_rows.items():
        with (output_dir / f"{split_name}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(split_data)

    print(f"Gambar berlabel dan ditinjau: {image_count}")
    print(f"Baris anotasi dilewati (kosong/belum ditinjau): {skipped}")
    for split_name in SPLITS:
        split_image_count = len({row["image_path"] for row in split_rows[split_name]})
        print(f"{split_name}: {split_image_count} gambar, {len(split_rows[split_name])} caption")
    print(f"Split tersimpan di: {output_dir}")


if __name__ == "__main__":
    main()
