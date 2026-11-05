"""Train a CNN-Transformer captioner in frozen-CNN and fine-tuning stages."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.cnn_captioner import ImageCaptioner, SPECIAL_TOKENS, image_transform


ROOT = Path(__file__).resolve().parent


def tokenize(text):
    return text.lower().split()


def build_vocabulary(rows, min_frequency=1):
    counts = Counter(token for row in rows for token in tokenize(row["caption"]))
    words = sorted(word for word, count in counts.items() if count >= min_frequency)
    vocab = list(SPECIAL_TOKENS) + [word for word in words if word not in SPECIAL_TOKENS]
    return {word: index for index, word in enumerate(vocab)}


def read_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if row.get("caption", "").strip()]
    if not rows:
        raise ValueError(f"Tidak ada anotasi ber-caption di {csv_path}")
    return rows


class ImageCaptionDataset(Dataset):
    def __init__(self, rows, vocabulary, max_words, augment=False):
        self.rows = rows
        self.vocabulary = vocabulary
        self.max_words = max_words
        self.transform = image_transform(224, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with Image.open(image_path) as opened:
            image = self.transform(opened.convert("RGB"))

        tokens = [self.vocabulary["<bos>"]]
        tokens.extend(self.vocabulary.get(word, self.vocabulary["<unk>"])
                      for word in tokenize(row["caption"])[:self.max_words])
        tokens.append(self.vocabulary["<eos>"])
        tokens += [self.vocabulary["<pad>"]] * (self.max_words + 2 - len(tokens))
        return image, torch.tensor(tokens, dtype=torch.long)


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        model.encoder.eval()
        if any(parameter.requires_grad for parameter in model.encoder.parameters()):
            model.encoder.features[-1].train()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_tokens = 0
    for images, tokens in loader:
        images, tokens = images.to(device), tokens.to(device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images, inputs)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
        valid_tokens = targets.ne(0).sum().item()
        total_loss += loss.detach().item() * valid_tokens
        total_tokens += valid_tokens
    return total_loss / max(total_tokens, 1)


def train_stage(model, train_loader, validation_loader, device, epochs, decoder_lr,
                encoder_lr, output_dir, stage_name, warmup_steps, patience):
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_parameters = list(model.decoder.parameters())
    groups = [{"params": decoder_parameters, "lr": decoder_lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(epochs * len(train_loader), 1)
    warmup_steps = min(warmup_steps, max(steps - 1, 0))

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max((steps - step) / max(steps - warmup_steps, 1), 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, scheduler)
        validation_loss = run_epoch(model, validation_loader, device)
        history.append({"stage": stage_name, "epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss})
        print(f"{stage_name} {epoch}/{epochs}: train={train_loss:.4f}, val={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale_epochs = 0
            torch.save({"model_state": best_state, "validation_loss": best_loss},
                       output_dir / f"best_{stage_name}.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping {stage_name} setelah {epoch} epoch.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, history


def main():
    parser = argparse.ArgumentParser(description="Latih model captioning CNN-Transformer dua tahap.")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--validation-csv", default="data/splits/validation.csv")
    parser.add_argument("--output-dir", default="outputs/cnn-transformer")
    parser.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--decoder", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--max-words", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_rows = read_rows((ROOT / args.train_csv).resolve())
    validation_rows = read_rows((ROOT / args.validation_csv).resolve())
    vocabulary = build_vocabulary(train_rows)
    max_length = args.max_words + 2
    model = ImageCaptioner(
        len(vocabulary), args.backbone, args.decoder, pretrained=True,
        d_model=args.d_model, decoder_layers=args.decoder_layers,
        attention_heads=args.attention_heads, max_length=max_length,
    ).to(device)
    train_data = ImageCaptionDataset(train_rows, vocabulary, args.max_words, args.augment)
    validation_data = ImageCaptionDataset(validation_rows, vocabulary, args.max_words)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device={device} | backbone={args.backbone} | decoder={args.decoder}")
    print(f"Train={len(train_rows)} caption | validation={len(validation_rows)} caption | vocab={len(vocabulary)}")
    history = []
    model.encoder.freeze()
    loss1, history1 = train_stage(
        model, train_loader, validation_loader, device, args.stage1_epochs,
        args.decoder_lr, args.encoder_lr, output_dir, "stage1_frozen",
        args.warmup_steps, args.patience,
    )
    history.extend(history1)
    model.encoder.unfreeze_last_block()
    loss2, history2 = train_stage(
        model, train_loader, validation_loader, device, args.stage2_epochs,
        args.decoder_lr, args.encoder_lr, output_dir, "stage2_finetune",
        args.warmup_steps, args.patience,
    )
    history.extend(history2)
    best_stage = "stage1_frozen" if loss1 <= loss2 else "stage2_finetune"
    chosen = torch.load(output_dir / f"best_{best_stage}.pt", map_location=device)
    model.load_state_dict(chosen["model_state"])
    config = {
        "backbone": args.backbone, "decoder": args.decoder,
        "vocabulary": vocabulary, "max_words": args.max_words,
        "d_model": args.d_model, "decoder_layers": args.decoder_layers,
        "attention_heads": args.attention_heads, "max_length": max_length,
        "best_stage": best_stage, "best_validation_loss": chosen["validation_loss"],
        "image_size": 224,
    }
    torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "model.pt")
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint akhir: {output_dir / 'model.pt'} (tahap terbaik: {best_stage})")


if __name__ == "__main__":
    main()
