import json
from pathlib import Path


def main() -> None:
    baseline = json.loads(Path("results/baseline_results.json").read_text())
    finetuned = json.loads(Path("results/finetuned_results.json").read_text())
    by_id = {item["sample_id"]: item for item in finetuned["predictions"]}
    rows = []
    for item in baseline["predictions"]:
        ft_item = by_id[item["sample_id"]]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "audio": item["audio"],
                "reference": item["reference"],
                "baseline_prediction": item["prediction"],
                "finetuned_prediction": ft_item["prediction"],
            }
        )
    result = {
        "baseline_model": baseline["model"],
        "finetuned_model": finetuned["model"],
        "baseline_wer": baseline["wer"],
        "finetuned_wer": finetuned["wer"],
        "rows": rows,
    }
    Path("results/comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
