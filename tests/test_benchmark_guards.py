import unittest
from pathlib import Path

from benchmark_guards import load_jsonl, score_guard, summarize, GuardAdapter


class BenchmarkGuardTests(unittest.TestCase):
    def test_load_jsonl_requires_text_and_label(self):
        temp_dir = Path(__file__).resolve().parent / "_tmp_benchmark"
        temp_dir.mkdir(exist_ok=True)
        path = temp_dir / "bench.jsonl"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.addCleanup(lambda: temp_dir.rmdir() if temp_dir.exists() and not any(temp_dir.iterdir()) else None)

        path.write_text('{"text":"hello","label":0}\n', encoding="utf-8")
        rows = load_jsonl(path)

        self.assertEqual(rows[0]["id"], "bench-0001")
        self.assertEqual(rows[0]["text"], "hello")

    def test_summarize_reports_classification_metrics(self):
        samples = [
            {"id": "a", "text": "safe", "label": 0},
            {"id": "b", "text": "attack", "label": 1},
        ]
        adapter = GuardAdapter("fake", lambda text: ("attack" in text, 0.9 if "attack" in text else 0.1))

        summary = summarize(score_guard(adapter, samples))

        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["tn"], 1)
        self.assertEqual(summary["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
