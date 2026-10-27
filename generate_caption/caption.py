"""Local inference helpers and command-line caption generation for fine-tuned BLIP."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "blip-finetuned"


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_caption_model(checkpoint=DEFAULT_CHECKPOINT, device="auto", local_files_only=True):
    checkpoint = str(checkpoint)
    runtime_device = choose_device(device)
    processor = BlipProcessor.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model.to(runtime_device)
    model.eval()
    return model, processor, runtime_device


def predict_caption(image, model, processor, device, max_new_tokens=48):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    else:
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Buat caption lokal memakai model BLIP fine-tuned.")
    parser.add_argument("image", help="Path gambar yang akan diberi caption")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    if not Path(args.checkpoint).is_dir():
        raise SystemExit(
            f"Checkpoint belum ditemukan: {args.checkpoint}. Jalankan train_captioner.py dahulu."
        )

    model, processor, device = load_caption_model(args.checkpoint, args.device)
    print(f"Perangkat: {device}")
    print(predict_caption(args.image, model, processor, device))


if __name__ == "__main__":
    main()
