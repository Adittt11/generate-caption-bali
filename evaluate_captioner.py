"""Evaluate a saved CNN captioner on a held-out test split."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from generate_caption.caption import load_caption_model, predict_caption


ROOT = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="Evaluasi CNN captioner pada split test terpisah.")
    parser.add_argument("--test-csv", default="data/splits/test.csv")
    parser.add_argument("--model", default="outputs/cnn-transformer")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--beam-size", type=int, default=3)
    args = parser.parse_args()

    test_csv = (ROOT / args.test_csv).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if not test_csv.is_file():
        raise SystemExit(f"File test tidak ditemukan: {test_csv}")
    examples = load_test_examples(test_csv)
    model, vocabulary, _transform, device, config = load_caption_model(args.model, args.device)
    id_to_word = {str(index): word for word, index in vocabulary.items()}

    predictions, reference_groups, result_rows = [], [], []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = []
    for index, (image_path_text, references) in enumerate(sorted(examples.items()), start=1):
        image_path = Path(image_path_text)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        prediction = predict_caption(
            image_path, model, vocabulary, id_to_word, device,
            beam_size=args.beam_size, max_words=config["max_words"],
        )
        predictions.append(prediction)
        reference_groups.append(references)
        rouge_scores.append(max(
            scorer.score(reference, prediction)["rougeL"].fmeasure for reference in references
        ))
        result_rows.append({
            "image_path": image_path_text,
            "prediction": prediction,
            "references": " || ".join(references),
        })
        print(f"[{index}/{len(examples)}] {image_path_text}: {prediction}")

    reference_streams = [
        [references[min(i, len(references) - 1)] for references in reference_groups]
        for i in range(max(map(len, reference_groups)))
    ]
    metrics = {
        "model": args.model,
        "backbone": config["backbone"],
        "decoder": config["decoder"],
        "device": str(device),
        "test_images": len(examples),
        "BLEU_4": BLEU(tokenize="none").corpus_score(predictions, reference_streams).score,
        "ROUGE_L_f1": sum(rouge_scores) / len(rouge_scores),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "prediction", "references"])
        writer.writeheader()
        writer.writerows(result_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
