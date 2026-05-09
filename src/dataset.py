"""
Dataset loading and management module.

Supports multiple benchmark datasets. Each dataset definition specifies
how to fetch and parse questions into a flat list of dicts.
"""

import json
import requests
from pathlib import Path
from typing import Optional

# Registry of supported datasets. Add new datasets here.
DATASETS = {
    "bullshit-v2": {
        "name": "BullshitBench v2",
        "url": "https://raw.githubusercontent.com/petergpt/bullshit-benchmark/main/questions.v2.json",
        "description": "100 nonsense prompts across 5 domains to test if models detect fabricated frameworks.",
        "local_cache": "data/cache/bullshit_v2.json",
        "parser": "bullshit",
    },
}


def list_available_datasets() -> dict:
    return {k: {"name": v["name"], "description": v["description"]} for k, v in DATASETS.items()}


def fetch_dataset(dataset_key: str, force_refresh: bool = False) -> dict:
    """Fetch a dataset, using local cache if available."""
    if dataset_key not in DATASETS:
        available = list(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {available}")

    ds_config = DATASETS[dataset_key]
    cache_path = Path(ds_config["local_cache"])

    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    response = requests.get(ds_config["url"], timeout=30)
    response.raise_for_status()
    data = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def extract_questions(raw_data: dict, dataset_key: str) -> list:
    """Parse the raw dataset JSON into a flat list of question dicts."""
    parser = DATASETS[dataset_key]["parser"]

    if parser == "bullshit":
        return _parse_bullshit(raw_data)

    raise ValueError(f"No parser defined for dataset '{dataset_key}'")


def _parse_bullshit(data: dict) -> list:
    """Extract questions from BullshitBench format (nested under techniques)."""
    questions = []
    for technique in data.get("techniques", []):
        for q in technique.get("questions", []):
            questions.append({
                "id": q.get("id"),
                "question": q.get("question"),
                "nonsensical_element": q.get("nonsensical_element", ""),
                "domain": q.get("domain", ""),
                "domain_group": q.get("domain_group", ""),
                "difficulty": q.get("difficulty", ""),
                "difficulty_label": q.get("difficulty_label", ""),
                "technique": q.get("technique", ""),
                "is_control": q.get("is_control", False),
            })
    return questions


def filter_questions(
    questions: list,
    domain_group: Optional[str] = None,
    technique: Optional[str] = None,
    exclude_controls: bool = False,
) -> list:
    """Filter questions by domain, technique, or control status."""
    result = questions
    if domain_group:
        result = [q for q in result if q.get("domain_group") == domain_group]
    if technique:
        result = [q for q in result if q.get("technique") == technique]
    if exclude_controls:
        result = [q for q in result if not q.get("is_control", False)]
    return result
