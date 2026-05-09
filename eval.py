"""Evaluate APID on the dataset validation split."""

from __future__ import annotations

import json
from pathlib import Path

from scanner import DEFAULT_DATASET_PATH, PromptInjectionScanner


def load_validation_rows(dataset_path: Path) -> list[dict[str, object]]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    return validation_rows or rows


def metric_summary(results: list[dict[str, object]]) -> dict[str, float | int]:
    tp = sum(1 for row in results if row["expected"] and row["predicted"])
    fp = sum(1 for row in results if not row["expected"] and row["predicted"])
    tn = sum(1 for row in results if not row["expected"] and not row["predicted"])
    fn = sum(1 for row in results if row["expected"] and not row["predicted"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0
    return {
        "samples": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def main() -> int:
    scanner = PromptInjectionScanner(use_transformer_embeddings=False)
    rows = load_validation_rows(DEFAULT_DATASET_PATH)
    results: list[dict[str, object]] = []

    for row in rows:
        result = scanner.scan(str(row["text"]))
        results.append(
            {
                "prompt": row["text"],
                "expected": bool(int(row["label"])),
                "predicted": bool(result["is_malicious"]),
                "score": result["risk_score"],
                "source": row.get("source", "unknown"),
            }
        )

    print(json.dumps({"summary": metric_summary(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
