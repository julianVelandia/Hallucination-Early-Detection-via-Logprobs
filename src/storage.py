"""
Persistent JSON storage for benchmark results.

Each entry is keyed by "{question_id}__{model}" so the same question
can be processed with multiple models without collision.
Skips already-processed entries on subsequent runs.
"""

import json
from pathlib import Path
from datetime import datetime


class ResultStorage:
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self._data: dict = None

    def _load(self) -> dict:
        if self._data is not None:
            return self._data
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {
                "metadata": {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "version": "1.0",
                    "description": "Logprob collection dataset for hallucination detection research",
                },
                "processed": {},
            }
        return self._data

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def is_processed(self, question_id: str, model: str) -> bool:
        data = self._load()
        key = _make_key(question_id, model)
        return key in data["processed"]

    def add_result(self, question_id: str, model: str, question_meta: dict, turns: list):
        """Persist a fully completed question (all turns) as a single entry."""
        data = self._load()
        key = _make_key(question_id, model)
        data["processed"][key] = {
            "question_id": question_id,
            "model": model,
            "question": question_meta.get("question"),
            "domain": question_meta.get("domain", ""),
            "domain_group": question_meta.get("domain_group", ""),
            "technique": question_meta.get("technique", ""),
            "difficulty": question_meta.get("difficulty", ""),
            "difficulty_label": question_meta.get("difficulty_label", ""),
            "nonsensical_element": question_meta.get("nonsensical_element", ""),
            "is_control": question_meta.get("is_control", False),
            "turns": turns,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        self._save()

    def get_processed_ids(self, model: str = None) -> set:
        """Return set of question_ids already processed (optionally for a specific model)."""
        data = self._load()
        ids = set()
        for key in data["processed"]:
            q_id, m = key.split("__", 1)
            if model is None or m == model:
                ids.add(q_id)
        return ids

    def get_stats(self, model: str = None) -> dict:
        data = self._load()
        all_entries = data["processed"]

        if model:
            entries = {k: v for k, v in all_entries.items() if k.endswith(f"__{model}")}
        else:
            entries = all_entries

        by_domain = {}
        by_technique = {}
        by_model: dict = {}
        for key, entry in entries.items():
            dg = entry.get("domain_group", "unknown")
            by_domain[dg] = by_domain.get(dg, 0) + 1
            t = entry.get("technique", "unknown")
            by_technique[t] = by_technique.get(t, 0) + 1
            m = entry.get("model", key.split("__", 1)[-1])
            by_model[m] = by_model.get(m, 0) + 1

        return {
            "total_entries": len(entries),
            "unique_questions": len({k.split("__")[0] for k in entries}),
            "by_model": by_model,
            "by_domain": by_domain,
            "by_technique": by_technique,
            "output_file": str(self.path),
        }

    def reset_question(self, question_id: str, model: str = None) -> int:
        data = self._load()
        keys_to_remove = []
        for k in data["processed"]:
            q_id, m = k.split("__", 1)
            if q_id == question_id and (model is None or m == model):
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del data["processed"][k]
        self._save()
        return len(keys_to_remove)

    def reset_all(self):
        self._data = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "version": "1.0",
                "description": "Logprob collection dataset for hallucination detection research",
            },
            "processed": {},
        }
        self._save()


def _make_key(question_id: str, model: str) -> str:
    return f"{question_id}__{model}"
