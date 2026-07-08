import csv
import json
from pathlib import Path
import wave


DATASET_DIR = Path("example_data/custom")
RESULTS_DIR = Path("results")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def read_split(split: str) -> list[dict]:
    text_path = DATASET_DIR / split / "text.csv"
    clips_dir = DATASET_DIR / split / "clips"
    with text_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    items = []
    for row in rows:
        audio_path = clips_dir / f"rec_{row['index']}.wav"
        items.append(
            {
                "index": int(row["index"]),
                "audio": str(audio_path),
                "sentence": row["sentence"],
                "duration_seconds": round(wav_duration(audio_path), 3),
            }
        )
    return items


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    splits = {"train": read_split("train"), "test": read_split("test")}
    summary = {
        "dataset": str(DATASET_DIR),
        "language": "English",
        "train_samples_available": len(splits["train"]),
        "test_samples_available": len(splits["test"]),
        "train_samples_used": 3,
        "validation_samples_used": 0,
        "test_samples_used": 2,
        "train_duration_seconds_available": round(
            sum(item["duration_seconds"] for item in splits["train"]), 3
        ),
        "test_duration_seconds_available": round(
            sum(item["duration_seconds"] for item in splits["test"]), 3
        ),
        "splits": splits,
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
