"""Benchmark APID against optional third-party prompt guard adapters."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from scanner import PromptInjectionScanner


BENCHMARK_DIR = Path(__file__).resolve().parent / "training_data" / "benchmarks"
DEFAULT_BENCHMARKS = (
    BENCHMARK_DIR / "public_guardrail_smoke.jsonl",
    BENCHMARK_DIR / "lakera_style.jsonl",
)


@dataclass
class Prediction:
    guard: str
    sample_id: str
    expected: bool
    predicted: bool
    score: float | None
    latency_ms: float
    skipped: bool = False
    error: str | None = None


@dataclass
class GuardAdapter:
    name: str
    predict: Callable[[str], tuple[bool, float | None]]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" not in row or "label" not in row:
                raise ValueError(f"{path}:{line_number} must include text and label")
            row.setdefault("id", f"{path.stem}-{line_number:04d}")
            row.setdefault("source", path.stem)
            samples.append(row)
    return samples


def load_samples(paths: Sequence[Path]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for path in paths:
        samples.extend(load_jsonl(path))
    return samples


def build_apid_adapter() -> GuardAdapter:
    scanner = PromptInjectionScanner(use_transformer_embeddings=False)

    def predict(text: str) -> tuple[bool, float | None]:
        result = scanner.scan(text)
        return bool(result["is_malicious"]), float(result["risk_score"])

    return GuardAdapter("apid", predict)


def build_llm_guard_adapter() -> GuardAdapter:
    try:
        from llm_guard.input_scanners.prompt_injection import PromptInjection
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"LLM Guard is unavailable: {exc}") from exc

    scanner = PromptInjection()

    def predict(text: str) -> tuple[bool, float | None]:
        _sanitized_prompt, is_valid, risk_score = scanner.scan(text)
        return not bool(is_valid), float(risk_score)

    return GuardAdapter("llm_guard", predict)


def build_llama_guard_adapter(model_id: str | None = None) -> GuardAdapter:
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Transformers is unavailable: {exc}") from exc

    model_id = model_id or os.getenv("APID_LLAMA_GUARD_MODEL", "meta-llama/Prompt-Guard-86M")
    classifier = pipeline("text-classification", model=model_id)

    def predict(text: str) -> tuple[bool, float | None]:
        output = classifier(text, truncation=True, max_length=512)
        row = output[0] if isinstance(output, list) else output
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        return label in {"injection", "jailbreak", "unsafe", "label_1", "label_2"}, score

    return GuardAdapter("llama_prompt_guard", predict)


def score_guard(adapter: GuardAdapter, samples: Iterable[dict[str, object]]) -> list[Prediction]:
    predictions: list[Prediction] = []
    for sample in samples:
        expected = bool(int(sample["label"]))
        started = time.perf_counter()
        try:
            predicted, score = adapter.predict(str(sample["text"]))
            error = None
            skipped = False
        except Exception as exc:
            predicted = False
            score = None
            error = str(exc)
            skipped = True
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            Prediction(
                guard=adapter.name,
                sample_id=str(sample["id"]),
                expected=expected,
                predicted=predicted,
                score=score,
                latency_ms=latency_ms,
                skipped=skipped,
                error=error,
            )
        )
    return predictions


def summarize(predictions: Sequence[Prediction]) -> dict[str, object]:
    active = [prediction for prediction in predictions if not prediction.skipped]
    skipped = len(predictions) - len(active)
    tp = sum(1 for row in active if row.expected and row.predicted)
    fp = sum(1 for row in active if not row.expected and row.predicted)
    tn = sum(1 for row in active if not row.expected and not row.predicted)
    fn = sum(1 for row in active if row.expected and not row.predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / len(active) if active else 0.0
    latencies = [row.latency_ms for row in active]
    return {
        "guard": predictions[0].guard if predictions else "unknown",
        "samples": len(predictions),
        "evaluated": len(active),
        "skipped": skipped,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else None,
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else None,
    }


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    return {
        "guard": prediction.guard,
        "sample_id": prediction.sample_id,
        "expected": prediction.expected,
        "predicted": prediction.predicted,
        "score": prediction.score,
        "latency_ms": round(prediction.latency_ms, 2),
        "skipped": prediction.skipped,
        "error": prediction.error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        type=Path,
        help="JSONL benchmark file. Defaults to all bundled benchmark fixtures.",
    )
    parser.add_argument("--include-llm-guard", action="store_true", help="Run Protect AI LLM Guard if installed.")
    parser.add_argument(
        "--include-llama-guard",
        action="store_true",
        help="Run a Transformers text-classification guard, defaulting to meta-llama/Prompt-Guard-86M.",
    )
    parser.add_argument("--llama-guard-model", help="Override the Transformers model used for the Llama/Prompt Guard adapter.")
    parser.add_argument("--json-output", type=Path, help="Write detailed benchmark results to this JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_paths = tuple(args.benchmark) if args.benchmark else DEFAULT_BENCHMARKS
    samples = load_samples(benchmark_paths)

    adapters: list[GuardAdapter] = [build_apid_adapter()]
    adapter_errors: dict[str, str] = {}

    if args.include_llm_guard:
        try:
            adapters.append(build_llm_guard_adapter())
        except RuntimeError as exc:
            adapter_errors["llm_guard"] = str(exc)

    if args.include_llama_guard:
        try:
            adapters.append(build_llama_guard_adapter(args.llama_guard_model))
        except RuntimeError as exc:
            adapter_errors["llama_prompt_guard"] = str(exc)

    all_predictions: list[Prediction] = []
    summaries: list[dict[str, object]] = []
    for adapter in adapters:
        predictions = score_guard(adapter, samples)
        all_predictions.extend(predictions)
        summaries.append(summarize(predictions))

    report = {
        "benchmarks": [str(path) for path in benchmark_paths],
        "adapter_errors": adapter_errors,
        "summaries": summaries,
        "predictions": [prediction_to_dict(row) for row in all_predictions],
    }

    print(json.dumps({"adapter_errors": adapter_errors, "summaries": summaries}, indent=2))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
