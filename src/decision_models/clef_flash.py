from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import QwenDecisionModel


class ClefFlashClient:
    """Lazy wrapper around Cloudflare's official Clef-Flash release code."""

    MODEL_ID = "Cloudflare/clef-flash"

    def __init__(self, model: Any, processor: Any, systemone_fn: Any) -> None:
        self.model = model
        self.processor = processor
        self._systemone = systemone_fn

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        *,
        device: str = "cuda",
        revision: str | None = None,
    ) -> "ClefFlashClient":
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, revision=revision))
        module_path = path / "joint_schema_model.py"
        spec = importlib.util.spec_from_file_location("clef_joint_schema_model", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model, processor = module.load_release_model(str(path), device=device)
        return cls(model, processor, module.systemone)

    def decide(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        response = self._systemone(
            self.model,
            self.processor,
            {"model": "clef-flash", "state": state, "questions": dict(questions)},
        )
        return response["answers"]

    def decide_many(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        return [self.decide(record["state"], record["questions"]) for record in records]

    def as_frozen_text_encoder(self, *, max_length: int = 512) -> QwenDecisionModel:
        """Expose Clef-Flash's Qwen3.5 backbone without its decision head."""
        language_model = self.model.language_model
        base_model = (
            language_model.get_base_model()
            if hasattr(language_model, "get_base_model")
            else language_model
        )
        text_model = base_model.model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        return QwenDecisionModel(
            text_model,
            self.processor.tokenizer,
            max_length=max_length,
            freeze_backbone=True,
        )
