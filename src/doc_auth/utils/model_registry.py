"""
Dataset versioning və model registry.

Məqsəd: hansı dataset versiyası ilə hansı model train olunub,
hansı metriklərlə nəticələnib — izlənə bilsin.

Sadə fayl-əsaslı implementasiya (production-da MLflow/DVC/W&B ilə əvəz edilə bilər,
bax: configs/model.yaml -> registry.tracking_backend).
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ModelRecord:
    model_id: str
    model_name: str
    dataset_version: str
    metrics: dict
    created_at: str
    params: dict = field(default_factory=dict)
    notes: str = ""


class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "index.json"
        if not self.index_file.exists():
            self.index_file.write_text("[]", encoding="utf-8")

    def register(
        self,
        model_name: str,
        dataset_version: str,
        metrics: dict,
        params: dict | None = None,
        notes: str = "",
    ) -> ModelRecord:
        record = ModelRecord(
            model_id=str(uuid.uuid4()),
            model_name=model_name,
            dataset_version=dataset_version,
            metrics=metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
            params=params or {},
            notes=notes,
        )
        records = self._load_index()
        records.append(record.__dict__)
        self.index_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    def _load_index(self) -> list[dict]:
        return json.loads(self.index_file.read_text(encoding="utf-8"))

    def list_models(self, model_name: str | None = None) -> list[dict]:
        records = self._load_index()
        if model_name:
            return [r for r in records if r["model_name"] == model_name]
        return records
