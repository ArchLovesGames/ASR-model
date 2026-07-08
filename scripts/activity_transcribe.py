import argparse
import csv
import json
import time
from pathlib import Path

import evaluate
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


DATASET_DIR = Path("example_data/custom")
RESULTS_DIR = Path("results")


def load_test_rows(limit: int) -> list[dict]:
    with (DATASET_DIR / "test" / "text.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    for row in rows:
        row["audio"] = str(DATASET_DIR / "test" / "clips" / f"rec_{row['index']}.wav")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = load_test_rows(args.limit)
    started = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
        device=-1,
    )
    load_seconds = time.time() - started

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")
    predictions = []
    infer_started = time.time()
    for row in rows:
        output = asr(row["audio"], generate_kwargs={"task": "transcribe"})
        predictions.append(
            {
                "sample_id": int(row["index"]),
                "audio": row["audio"],
                "reference": row["sentence"],
                "prediction": output["text"],
            }
        )

    references_norm = [normalizer(item["reference"]) for item in predictions]
    predictions_norm = [normalizer(item["prediction"]) for item in predictions]
    result = {
        "model": args.model,
        "num_samples": len(predictions),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(time.time() - infer_started, 3),
        "wer": 100 * wer_metric.compute(predictions=predictions_norm, references=references_norm),
        "predictions": predictions,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
