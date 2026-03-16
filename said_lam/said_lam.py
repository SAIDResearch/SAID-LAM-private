"""SAID-LAM model for MTEB.

Install: pip install said-lam
Weights: https://huggingface.co/SAIDResearch/SAID-LAM-v1

Organization: SAIDResearch
Reference: https://saidhome.ai
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        Array,
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        PromptType,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

logger = logging.getLogger(__name__)


def said_lam_loader(
    model_name: str, revision: str | None = None, device: str | None = None, **kwargs
):
    """Load SAID-LAM for MTEB evaluation. Returns LAM (same class as normal use)."""
    from said_lam import LAM
    model = LAM(model_name, device=device, **kwargs)
    model.auto_activate_mteb()
    return model


def _extract_texts(inputs) -> list[str]:
    """Extract texts from MTEB DataLoader batches."""
    texts = []
    for batch in inputs:
        if isinstance(batch, dict) and "text" in batch:
            v = batch["text"]
            texts.extend(v if isinstance(v, list) else [str(v)])
        elif isinstance(batch, (list, tuple)):
            texts.extend(str(s) for s in batch)
        else:
            texts.append(str(batch))
    return texts


# =============================================================================
# Model Registration — public metadata only (same as HuggingFace model card)
# =============================================================================

said_lam_v1 = ModelMeta(
    name="SAIDResearch/SAID-LAM-v1",
    revision="main",
    release_date="2026-01-01",
    languages=["eng"],
    loader=said_lam_loader,
    n_parameters=23_848_788,
    memory_usage_mb=90,
    max_tokens=32768,
    embed_dim=384,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    modalities=["text"],
    framework=["PyTorch"],
    reference="https://saidhome.ai",
    similarity_fn_name="cosine",
    use_instructions=False,
)
