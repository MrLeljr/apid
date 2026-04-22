"""Core prompt-injection detection logic."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence

import numpy as np
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "training_data" / "prompt_injection_dataset.json"
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "scanner.joblib"
RESET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bignore (all|any|the) previous (instructions|messages|context)\b",
        r"\bforget (all|any|the) previous (instructions|messages|context)\b",
        r"\breset (the )?(conversation|chat|context|memory)\b",
        r"\bstart over\b",
        r"\bnew instructions\b",
        r"\bfrom now on\b",
        r"\byou are now\b",
        r"\bdisregard (all|any|the) (earlier|prior|previous) (instructions|messages|context)\b",
    )
)
PRIOR_MESSAGE_REFERENCE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bprevious messages\b",
        r"\bearlier messages\b",
        r"\bprior messages\b",
        r"\babove conversation\b",
        r"\bchat history\b",
        r"\bconversation history\b",
        r"\bwhat did i say earlier\b",
        r"\bas mentioned above\b",
    )
)
CONTRADICTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bignore the system prompt\b",
        r"\boverride (your|all) instructions\b",
        r"\bbypass (safety|policy|guardrails|filters)\b",
        r"\bdo not follow (the )?(previous|prior|earlier) instructions\b",
        r"\bpretend (that )?(the )?(previous|prior) instructions do not exist\b",
    )
)
CONTEXT_CONFLICT_PATTERN = re.compile(r"\b(ignore|forget|disregard)\b")


class PromptInjectionScanner:
    """Detect prompt-injection attempts with classifier, semantic, and rule-based layers."""

    def __init__(
        self,
        decision_threshold: float = 0.55,
        semantic_similarity_threshold: float = 0.62,
        semantic_margin_threshold: float = 0.08,
        semantic_top_k: int = 3,
        dataset_path: Optional[str] = None,
        artifact_path: Optional[str] = None,
        model_name: str = "paraphrase-mpnet-base-v2",
        use_transformer_embeddings: bool = True,
    ):
        """Initialize the scanner and eagerly train it from the local dataset."""

        self.decision_threshold = decision_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.semantic_margin_threshold = semantic_margin_threshold
        self.semantic_top_k = semantic_top_k
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
        self.preferred_model_name = model_name
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.label_names = {0: "benign", 1: "prompt_injection"}
        self.training_examples: List[Dict[str, str | int]] = []
        self.training_texts: List[str] = []
        self.training_labels: List[int] = []
        self.training_embeddings: np.ndarray | None = None
        self._malicious_indices = np.array([], dtype=int)
        self._benign_indices = np.array([], dtype=int)
        self._is_trained = False
        self.embedding_backend = "hashing_vectorizer"
        self.vectorizer = HashingVectorizer(
            n_features=2048,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
        )

        self.model = None
        if use_transformer_embeddings:
            try:
                self.model = SentenceTransformer(model_name, local_files_only=True)
                self.embedding_backend = f"sentence_transformer:{model_name}"
            except Exception:
                self.model = None

        if not self.load_artifacts():
            self.train_from_file(self.dataset_path)

    def load_examples(self, dataset_path: Path) -> List[Dict[str, str | int]]:
        """Load and validate labeled samples from disk."""

        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            payload = json.load(dataset_file)

        if not isinstance(payload, list):
            raise ValueError("Training dataset must be a list of {'text', 'label'} records.")

        examples: List[Dict[str, str | int]] = []
        for row in payload:
            text = str(row.get("text", "")).strip()
            label = int(row.get("label", 0))
            if not text:
                continue
            if label not in (0, 1):
                raise ValueError("Labels must be 0 (benign) or 1 (prompt injection).")
            example = dict(row)
            example["text"] = text
            example["label"] = label
            examples.append(example)

        if len(examples) < 2:
            raise ValueError("Training dataset must contain at least two labeled examples.")

        labels = {example["label"] for example in examples}
        if labels != {0, 1}:
            raise ValueError("Training dataset must include both benign and malicious samples.")

        return examples

    def train(self, examples: Sequence[Dict[str, str | int]]) -> None:
        """Train the classifier and cache embeddings for semantic comparison."""

        texts = [str(example["text"]) for example in examples]
        labels = [int(example["label"]) for example in examples]

        embeddings = self._embed_texts(texts)
        self.classifier.fit(embeddings, labels)

        self.training_examples = list(examples)
        self.training_texts = texts
        self.training_labels = labels
        self.training_embeddings = embeddings
        label_array = np.asarray(labels)
        self._malicious_indices = np.flatnonzero(label_array == 1)
        self._benign_indices = np.flatnonzero(label_array == 0)
        self._is_trained = True
        self.save_artifacts()

    def train_from_file(self, dataset_path: Path) -> None:
        """Train the scanner from a JSON dataset file."""

        self.train(self.load_examples(dataset_path))

    def _artifact_metadata(self) -> Dict[str, object]:
        """Describe the training inputs so cached artifacts can be validated."""

        dataset_stat = self.dataset_path.stat()
        return {
            "dataset_path": str(self.dataset_path.resolve()),
            "dataset_mtime_ns": dataset_stat.st_mtime_ns,
            "decision_threshold": self.decision_threshold,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "semantic_margin_threshold": self.semantic_margin_threshold,
            "semantic_top_k": self.semantic_top_k,
            "preferred_model_name": self.preferred_model_name,
            "embedding_backend": self.embedding_backend,
        }

    def save_artifacts(self) -> None:
        """Persist the trained classifier and cached embeddings for faster startup."""

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "metadata": self._artifact_metadata(),
                "classifier": self.classifier,
                "vectorizer": self.vectorizer,
                "label_names": self.label_names,
                "training_examples": self.training_examples,
                "training_texts": self.training_texts,
                "training_labels": self.training_labels,
                "training_embeddings": self.training_embeddings,
                "malicious_indices": self._malicious_indices,
                "benign_indices": self._benign_indices,
            },
            self.artifact_path,
        )

    def load_artifacts(self) -> bool:
        """Load cached artifacts when they match the current dataset and thresholds."""

        if not self.artifact_path.exists() or not self.dataset_path.exists():
            return False

        try:
            payload = load(self.artifact_path)
        except Exception:
            return False

        if payload.get("metadata") != self._artifact_metadata():
            return False

        self.classifier = payload["classifier"]
        self.vectorizer = payload["vectorizer"]
        self.label_names = payload["label_names"]
        self.training_examples = list(payload["training_examples"])
        self.training_texts = list(payload["training_texts"])
        self.training_labels = list(payload["training_labels"])
        self.training_embeddings = np.asarray(payload["training_embeddings"], dtype=np.float32)
        self._malicious_indices = np.asarray(payload["malicious_indices"], dtype=int)
        self._benign_indices = np.asarray(payload["benign_indices"], dtype=int)
        self._is_trained = True
        return True

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts with the preferred backend and normalize dtypes."""

        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def _top_scores(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Return the highest scores for a label slice without fully sorting the array."""

        if indices.size == 0:
            return np.array([], dtype=np.float32)

        subset = values[indices]
        top_k = min(self.semantic_top_k, subset.size)
        if subset.size <= top_k:
            return np.sort(subset)[::-1]

        partitioned = np.partition(subset, subset.size - top_k)[-top_k:]
        return np.sort(partitioned)[::-1]

    def _rank_neighbors(self, prompt_embedding: np.ndarray) -> tuple[np.ndarray, List[Dict[str, object]]]:
        """Find the nearest training examples to the input prompt."""

        similarities = self.training_embeddings @ prompt_embedding
        top_k = min(self.semantic_top_k, similarities.size)
        ranked_indices = np.argpartition(similarities, similarities.size - top_k)[-top_k:]
        ranked_indices = ranked_indices[np.argsort(similarities[ranked_indices])[::-1]]

        neighbors: List[Dict[str, object]] = []
        for idx in ranked_indices:
            neighbors.append(
                {
                    "text": self.training_texts[idx],
                    "label": self.label_names[self.training_labels[idx]],
                    "source": self.training_examples[idx].get("source", "local"),
                    "similarity": round(float(similarities[idx]), 4),
                }
            )

        return similarities, neighbors

    def _semantic_layer(self, similarities: np.ndarray, neighbors: List[Dict[str, object]]) -> Dict[str, object]:
        """Compare the prompt against the most similar benign and malicious examples."""

        malicious_neighbors = [neighbor for neighbor in neighbors if neighbor["label"] == "prompt_injection"]
        benign_neighbors = [neighbor for neighbor in neighbors if neighbor["label"] == "benign"]
        malicious_scores = self._top_scores(similarities, self._malicious_indices)
        benign_scores = self._top_scores(similarities, self._benign_indices)

        top_malicious_similarity = float(malicious_scores[0]) if malicious_scores.size else 0.0
        top_benign_similarity = float(benign_scores[0]) if benign_scores.size else 0.0
        malicious_centroid_similarity = float(np.mean(malicious_scores)) if malicious_scores.size else 0.0
        benign_centroid_similarity = float(np.mean(benign_scores)) if benign_scores.size else 0.0
        semantic_margin = malicious_centroid_similarity - benign_centroid_similarity

        semantic_hit = (
            top_malicious_similarity >= self.semantic_similarity_threshold
            and semantic_margin >= self.semantic_margin_threshold
        )
        semantic_score = max(
            top_malicious_similarity,
            min(1.0, max(0.0, (semantic_margin + 1.0) / 2.0)),
        )

        return {
            "triggered": semantic_hit,
            "score": round(float(semantic_score), 4),
            "top_malicious_similarity": round(top_malicious_similarity, 4),
            "top_benign_similarity": round(top_benign_similarity, 4),
            "malicious_centroid_similarity": round(malicious_centroid_similarity, 4),
            "benign_centroid_similarity": round(benign_centroid_similarity, 4),
            "similarity_margin": round(semantic_margin, 4),
            "similarity_threshold": self.semantic_similarity_threshold,
            "margin_threshold": self.semantic_margin_threshold,
            "top_neighbors": neighbors,
            "matched_signal": malicious_neighbors[0]["text"] if malicious_neighbors else None,
            "benign_counterexample": benign_neighbors[0]["text"] if benign_neighbors else None,
        }

    def _append_pattern_matches(
        self,
        prompt: str,
        patterns: Sequence[Pattern[str]],
        label: str,
        score_delta: float,
        matched_signals: List[str],
        score: float,
    ) -> float:
        """Accumulate matches for the rule-based behavioral layer."""

        for pattern in patterns:
            if pattern.search(prompt):
                matched_signals.append(f"{label}:{pattern.pattern}")
                score += score_delta
        return score

    def _behavioral_context_layer(self, prompt: str, context: Optional[str] = None) -> Dict[str, object]:
        """Apply lightweight contextual heuristics to catch instruction overrides."""

        prompt_lower = prompt.lower()
        matched_signals: List[str] = []
        score = 0.0

        score = self._append_pattern_matches(
            prompt_lower,
            RESET_PATTERNS,
            "reset_attempt",
            0.35,
            matched_signals,
            score,
        )

        suspicious_history_reference = False
        for pattern in PRIOR_MESSAGE_REFERENCE_PATTERNS:
            if pattern.search(prompt_lower):
                suspicious_history_reference = context is None or context.strip() == ""
                if suspicious_history_reference:
                    matched_signals.append(f"suspicious_history_reference:{pattern.pattern}")
                    score += 0.25

        score = self._append_pattern_matches(
            prompt_lower,
            CONTRADICTION_PATTERNS,
            "instruction_override",
            0.35,
            matched_signals,
            score,
        )

        if context and CONTEXT_CONFLICT_PATTERN.search(prompt_lower):
            matched_signals.append("context_conflict:prompt_attempts_to_override_existing_context")
            score += 0.2

        score = min(1.0, score)
        triggered = score >= 0.4

        return {
            "name": "Behavioral/contextual heuristics",
            "model": "rule_based_context_awareness",
            "triggered": triggered,
            "score": round(score, 4),
            "history_awareness_flag": bool(matched_signals),
            "suspicious_history_reference": suspicious_history_reference,
            "matched_signals": matched_signals,
        }

    def scan(self, prompt: str, context: Optional[str] = None) -> Dict[str, object]:
        """Score a prompt and return the full multi-layer detection report."""

        if not self._is_trained:
            raise RuntimeError("Scanner is not trained. Load a labeled dataset before scanning.")

        full_text = f"{context}\n{prompt}" if context else prompt
        prompt_embedding = self._embed_texts([full_text])[0]

        malicious_probability = float(self.classifier.predict_proba([prompt_embedding])[0][1])
        classifier_hit = malicious_probability >= self.decision_threshold
        similarities, neighbors = self._rank_neighbors(prompt_embedding)
        semantic_layer = self._semantic_layer(similarities, neighbors)
        behavioral_layer = self._behavioral_context_layer(prompt=prompt, context=context)
        is_malicious = classifier_hit or bool(semantic_layer["triggered"]) or bool(behavioral_layer["triggered"])
        final_score = max(
            malicious_probability,
            float(semantic_layer["score"]),
            float(behavioral_layer["score"]),
        )

        return {
            "prompt_preview": prompt[:250] + "..." if len(prompt) > 250 else prompt,
            "is_malicious": is_malicious,
            "risk_score": round(final_score, 4),
            "severity": "HIGH" if final_score > 0.85 else "MEDIUM" if final_score > 0.6 else "LOW",
            "layers": {
                "layer_1_classifier": {
                    "name": "Embedding classifier",
                    "model": f"{self.embedding_backend}_logistic_regression",
                    "triggered": classifier_hit,
                    "score": round(malicious_probability, 4),
                    "threshold": self.decision_threshold,
                },
                "layer_2_semantic": {
                    "name": "Semantic similarity",
                    "model": self.embedding_backend,
                    **semantic_layer,
                },
                "layer_3_behavioral_contextual": behavioral_layer,
            },
            "explanation": {
                "model": "multi_layer_detection",
                "preferred_model": self.preferred_model_name,
                "classifier_threshold": self.decision_threshold,
                "semantic_similarity_threshold": self.semantic_similarity_threshold,
                "semantic_margin_threshold": self.semantic_margin_threshold,
                "dataset_path": str(self.dataset_path),
                "training_samples": len(self.training_examples),
                "decision_strategy": "block_if_any_layer_triggers",
                "matched_signal": semantic_layer["matched_signal"],
                "benign_counterexample": semantic_layer["benign_counterexample"],
                "top_neighbors": semantic_layer["top_neighbors"],
                "history_awareness_flag": behavioral_layer["history_awareness_flag"],
                "behavioral_signals": behavioral_layer["matched_signals"],
            },
            "recommendation": "BLOCK" if is_malicious else "ALLOW",
        }
