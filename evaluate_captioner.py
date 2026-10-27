import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from transformers import BlipForConditionalGeneration, BlipProcessor


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Salesforce/blip-image-captioning-base"


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_test_examples(csv_path):
    grouped = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            caption = (row.get("caption") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if caption and image_path:
                grouped[image_path].append(caption)
    if not grouped:
        raise ValueError(f"Tidak ada caption referensi di {csv_path}")
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Evaluasi captioner pada split test yang terpisah.")
    parser.add_argument("--test-csv", default="data/splits/test.csv")
    parser.add_argument("--model", default="outputs/blip-finetuned")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    test_csv = (ROOT / args.test_csv).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if not test_csv.is_file():
        raise SystemExit(f"File test tidak ditemukan: {test_csv}")

    examples = load_test_examples(test_csv)
    device = choose_device(args.device)
    local_model = Path(args.model).exists()
    processor = BlipProcessor.from_pretrained(args.model, local_files_only=local_model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, local_files_only=local_model)
    model.to(device)
    model.eval()

    predictions = []
    reference_groups = []
    result_rows = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = []

    for index, (image_path_text, references) in enumerate(sorted(examples.items()), start=1):
        image_path = Path(image_path_text)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=3,
            )
        prediction = processor.decode(output_ids[0], skip_special_tokens=True).strip()

        predictions.append(prediction)
        reference_groups.append(references)
        rouge_scores.append(max(
            scorer.score(reference, prediction)["rougeL"].fmeasure
            for reference in references
        ))
        result_rows.append({
            "image_path": image_path_text,
            "prediction": prediction,
            "references": " || ".join(references),
        })
        print(f"[{index}/{len(examples)}] {image_path_text}: {prediction}")

    max_references = max(len(references) for references in reference_groups)
    reference_streams = [
        [references[min(reference_index, len(references) - 1)] for references in reference_groups]
        for reference_index in range(max_references)
    ]
    bleu_score = BLEU(tokenize="none").corpus_score(predictions, reference_streams).score
    metrics = {
        "model": args.model,
        "device": str(device),
        "test_images": len(examples),
        "BLEU_4": bleu_score,
        "ROUGE_L_f1": sum(rouge_scores) / len(rouge_scores),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "prediction", "references"])
        writer.writeheader()
        writer.writerows(result_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
