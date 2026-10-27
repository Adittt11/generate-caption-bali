import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Salesforce/blip-image-captioning-base"


def read_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if row.get("caption", "").strip()]
    if not rows:
        raise ValueError(f"Tidak ada anotasi ber-caption di {csv_path}")
    return rows


class ImageCaptionDataset(Dataset):
    def __init__(self, rows, processor, max_length):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        encoded = self.processor(
            images=image,
            text=row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(**batch)
            loss = outputs.loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = batch["pixel_values"].shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BLIP untuk caption budaya Bali.")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--validation-csv", default="data/splits/validation.csv")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="outputs/blip-finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_csv = (ROOT / args.train_csv).resolve()
    validation_csv = (ROOT / args.validation_csv).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if not train_csv.is_file() or not validation_csv.is_file():
        raise SystemExit("CSV train/validation belum ada. Jalankan prepare_dataset.py dahulu.")

    train_rows = read_rows(train_csv)
    validation_rows = read_rows(validation_csv)
    processor = BlipProcessor.from_pretrained(args.base_model)
    model = BlipForConditionalGeneration.from_pretrained(args.base_model)
    device = choose_device(args.device)
    model.to(device)

    train_data = ImageCaptionDataset(train_rows, processor, args.max_length)
    validation_data = ImageCaptionDataset(validation_rows, processor, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    history = []
    print(f"Model awal: {args.base_model}")
    print(f"Perangkat: {device}; data train: {len(train_rows)} caption; validation: {len(validation_rows)} caption")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            (output_dir / "training_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Checkpoint terbaik disimpan: {output_dir}")

    print(f"Fine-tuning selesai. Validation loss terbaik: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
