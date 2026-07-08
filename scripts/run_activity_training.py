import json
import time
from pathlib import Path

from speech_to_text_finetune.finetune_whisper import run_finetuning


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    started = time.time()
    baseline, finetuned = run_finetuning("activity_config.yaml")
    result = {
        "config": "activity_config.yaml",
        "model_checkpoint": "openai/whisper-tiny",
        "output_model_dir": "artifacts/activity-whisper-tiny-custom",
        "dataset": "example_data/custom",
        "train_samples": 3,
        "test_samples": 2,
        "runtime_seconds": round(time.time() - started, 3),
        "baseline_eval": baseline,
        "finetuned_eval": finetuned,
    }
    Path("results/evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
