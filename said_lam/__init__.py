"""
SAID-LAM: Linear Attention Model
384-dimensional embeddings | O(n) complexity

Usage:
    from said_lam import LAM
    model = LAM("SAIDResearch/SAID-LAM-v1")
    embeddings = model.encode(["Hello world"])

    # MTEB evaluation (encode + index/search via mteb.evaluate)
    import mteb
    tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "STS12"])
    results = mteb.evaluate(model=model, tasks=tasks)
"""
from __future__ import annotations

import logging
import numpy as np
import threading
from typing import List, Optional, Tuple
from pathlib import Path

__version__ = "1.0.3"
__author__ = "SAIDResearch"

logger = logging.getLogger(__name__)

# ============================================================================
# RUST BACKEND — must initialize CUDA *before* any torch/mteb import
# ============================================================================

try:
    import lam_candle
    _HAS_CANDLE = True
except ImportError:
    _HAS_CANDLE = False
    raise ImportError("lam_candle not found - pip install said-lam")

# Eagerly initialize CUDA + cuBLAS BEFORE torch gets imported (by mteb below).
# PyTorch's CUDA context initialization corrupts cudarc's cuBLAS handle;
# warming up first ensures the handle is created in a clean state.
try:
    lam_candle.cuda_warmup()
except Exception:
    pass

# MTEB loader + metadata (re-exported) — imports mteb which imports torch
from .said_lam import said_lam_loader, said_lam_v1  # noqa: F401

# Optional: LAM can serve as MTEB encoder (same class for all users)
try:
    from mteb.models.abs_encoder import AbsEncoder as _AbsEncoder
except ImportError:
    _AbsEncoder = object

# Export tier constants
TIER_FREE = lam_candle.TIER_FREE
TIER_BETA = lam_candle.TIER_BETA
TIER_LICENSED = lam_candle.TIER_LICENSED
TIER_INFINITE = lam_candle.TIER_INFINITE


# ============================================================================
# LAM CLASS
# ============================================================================

def _extract_texts_mteb(inputs):
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


class LAM(_AbsEncoder):
    """
    LAM (Linear Attention Models) — SAID-LAM-v1: Linear Attention Memory

    One class for embedding and MTEB evaluation.

    Public API:
        encode(sentences)  - Embed texts (up to 12K tokens per text → one embedding).
        For retrieval (index/search), use via mteb.evaluate(model=model, tasks=...).
    """

    def __init__(
        self,
        model_name_or_path: str = "SAIDResearch/SAID-LAM-v1",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        backend: str = "crystalline",
        **kwargs,
    ):
        self.model_name = model_name_or_path
        self._embedding_dim = 384
        self._output_dim = kwargs.pop("output_dim", None)

        # Backend: "crystalline" (LamEngine + CrystallineCore)
        self._backend = (backend or "crystalline").lower().strip()
        if self._backend != "crystalline":
            raise ValueError(f"backend must be 'crystalline', got {backend!r}")

        # Device selection: Rust auto-detects (CUDA if available, else CPU)
        import os
        if device and device.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            # Candle/CUDA device selection can be surprising on multi-GPU hosts or when
            # running inside containers. If the user didn't specify a mapping, default
            # to GPU 0 for stable behavior.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self._device = device or "cuda"

        # CUDA/cuBLAS initialization and handles can be sensitive to concurrent calls.
        # MTEB evaluators (notably STS) may call encode() from multiple threads.
        # We serialize all Rust engine calls to keep GPU execution stable.
        self._engine_lock = threading.RLock()

        # Find model path
        model_path = self._resolve_model_path(model_name_or_path)

        # Create Rust engine
        self._engine = lam_candle.LamEngine(str(model_path), backend=self._backend)

        # Accumulator for batch indexing via index_mteb()
        self._pending_ids: List[str] = []
        self._pending_texts: List[str] = []
        self._index_built = False

        if self._engine.has_model():
            logger.info(f"LAM loaded | Tier: {self._engine.get_tier_name()} | Backend: {self._backend}")
        else:
            raise RuntimeError(f"Failed to load model from {model_path}")

        try:
            from said_lam.said_lam import said_lam_v1
            self.mteb_model_meta = said_lam_v1
        except ImportError:
            self.mteb_model_meta = None

    def _resolve_model_path(self, name_or_path: str) -> Path:
        """Find model weights.

        Search order:
        1. Direct path (local directory)
        2. 'weights/' directory relative to package or working directory
        3. /workspace/LAM/HuggingFace_Submission/SAID-LAM-v1 (workspace)
        4. HuggingFace cache (already downloaded)
        5. Download from HuggingFace Hub
        """
        path = Path(name_or_path)

        # Direct local path
        if path.exists():
            if path.is_file():
                return path.parent
            if (path / "model.safetensors").exists():
                return path

        # Check for weights/ directory relative to package or cwd (dev/testing)
        for base in [Path(__file__).parent.parent, Path.cwd()]:
            weights_dir = base / "weights"
            if weights_dir.exists() and (weights_dir / "model.safetensors").exists():
                return weights_dir

        # Local hf_model_card (repo root or cwd) — use before HF when repo not yet published
        for base in [Path(__file__).parent.parent, Path.cwd(), Path.cwd().parent]:
            hf_card = base / "hf_model_card"
            if hf_card.exists() and (hf_card / "model.safetensors").exists():
                return hf_card

        # Workspace fallback
        workspace = Path("/workspace/LAM/HuggingFace_Submission/SAID-LAM-v1")
        if workspace.exists() and (workspace / "model.safetensors").exists():
            return workspace

        # HuggingFace cache (try both SAIDResearch and Said-Research for backwards compat)
        for repo_id in (name_or_path, name_or_path.replace("Said-Research", "SAIDResearch"), name_or_path.replace("SAIDResearch", "Said-Research")):
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
            if cache_dir.exists():
                snapshots = cache_dir / "snapshots"
                if snapshots.exists():
                    for snapshot in sorted(snapshots.iterdir(), reverse=True):
                        if snapshot.is_dir() and (snapshot / "model.safetensors").exists():
                            return snapshot

        # Download from HuggingFace Hub (canonical org: SAIDResearch)
        try:
            from huggingface_hub import snapshot_download
            repo_id = name_or_path.replace("Said-Research", "SAIDResearch")
            downloaded = snapshot_download(repo_id)
            if (Path(downloaded) / "model.safetensors").exists():
                return Path(downloaded)
        except ImportError:
            raise ImportError(
                "Model weights not found locally. Install huggingface_hub to download:\n"
                "  pip install huggingface_hub\n"
                "Then: from said_lam import LAM; model = LAM()"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model weights from HuggingFace: {e}\n"
                f"Model: {name_or_path}\n"
                "Ensure you have internet access or provide a local path."
            )

        return path

    # ========================================================================
    # TIER MANAGEMENT
    # ========================================================================

    def activate(self, activation_key: str) -> bool:
        """Activate a higher tier to unlock SCA (Said Crystalline Attention)."""
        return self._engine.activate(activation_key)

    def register_beta(self, email: str = "") -> bool:
        """Register for a free 1-month beta key (32K tokens, SCA enabled)."""
        return self._engine.register_beta(email)

    def request_another_beta(self, email: str = "") -> str:
        """Request another beta trial after expiry."""
        return self._engine.request_another_beta(email)

    @property
    def tier(self) -> str:
        """Current tier name (FREE, BETA, LICENSED, INFINITE)."""
        try:
            return self._engine.get_tier_name()
        except AttributeError:
            try:
                level = self._engine.get_tier()
                return {0: "FREE", 1: "BETA", 2: "LICENSED", 3: "INFINITE"}.get(level, "UNKNOWN")
            except AttributeError:
                return "FREE"

    @property
    def max_tokens(self) -> int:
        """Maximum token limit for current tier."""
        try:
            return self._engine.get_max_tokens()
        except AttributeError:
            return 12000

    def auto_activate_mteb(self) -> bool:
        """Auto-activate for MTEB evaluation (unlocks full capability)."""
        return self._engine.auto_activate_mteb()

    def clear(self) -> None:
        """Clear all indexed documents and pending accumulator."""
        self._engine.clear()
        self._pending_ids.clear()
        self._pending_texts.clear()
        self._index_built = False

    # ========================================================================
    # CORE: encode(), index(), search()
    # ========================================================================

    def encode(
        self,
        sentences=None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        doc_ids: Optional[List[str]] = None,
        store_for_recall: Optional[bool] = None,
        output_dim: Optional[int] = None,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts to embeddings. Simple API: up to 12K tokens per text → one embedding. MTEB: DataLoader + task_metadata."""
        # MTEB protocol: encode(inputs=DataLoader, task_metadata=..., ...)
        if task_metadata is not None:
            # Important: do NOT materialize the whole DataLoader into one mega-list.
            # Some MTEB evaluators can pass very large iterables; encoding in smaller
            # chunks is both faster (less peak RAM) and more stable on CUDA backends.
            out_chunks: list[np.ndarray] = []
            total = 0
            for batch in sentences:
                if isinstance(batch, dict) and "text" in batch:
                    v = batch["text"]
                    texts = v if isinstance(v, list) else [str(v)]
                elif isinstance(batch, (list, tuple)):
                    texts = [str(s) for s in batch]
                else:
                    texts = [str(batch)]
                if not texts:
                    continue
                total += len(texts)
                with self._engine_lock:
                    arr = self._engine.encode(
                        texts, normalize_embeddings, batch_size, doc_ids, store_for_recall
                    )
                out_chunks.append(np.asarray(arr, dtype=np.float32))

            if total == 0:
                return np.zeros((0, self._embedding_dim), dtype=np.float32)

            result = np.concatenate(out_chunks, axis=0) if len(out_chunks) > 1 else out_chunks[0]
            odim = output_dim if output_dim is not None else self._output_dim
            if odim is not None and odim < self._embedding_dim:
                with self._engine_lock:
                    result = self._engine.truncate_embeddings(result, odim)
            return result

        # Simple API: encode(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]
        if sentences is None:
            sentences = []
        if not sentences:
            dim = output_dim or self._embedding_dim
            return np.zeros((0, dim), dtype=np.float32)

        with self._engine_lock:
            embeddings = self._engine.encode(
                sentences,
                normalize_embeddings,
                batch_size,
                doc_ids,
                store_for_recall,
            )
        result = np.array(embeddings, dtype=np.float32)
        if output_dim is not None and output_dim < self._embedding_dim:
            with self._engine_lock:
                result = self._engine.truncate_embeddings(result, output_dim)
        return result

    def index(
        self,
        doc_id_or_corpus=None,
        text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        encode_kwargs=None,
        num_proc=None,
        **kwargs,
    ) -> None:
        """Index a document or corpus. Use via mteb.evaluate() for retrieval; or index(doc_id, text) for direct use."""
        # MTEB protocol: index(corpus=..., task_metadata=..., ...) — all keyword args
        if task_metadata is not None:
            corpus = doc_id_or_corpus if doc_id_or_corpus is not None else kwargs.get("corpus")
            if corpus is None:
                raise ValueError("index() for MTEB requires corpus (positional or keyword)")
            task = str(getattr(task_metadata, "name", ""))
            ids_list, texts_list = [], []
            for doc in corpus:
                ids_list.append(str(doc["id"]))
                title = doc.get("title", "") or ""
                t = doc.get("text", "") or ""
                texts_list.append(f"{title} {t}".strip() if title else t)
            with self._engine_lock:
                self._engine.auto_activate_mteb()
                self._engine.index_mteb(ids_list, texts_list, task, None)
            return

        # Simple API: index(doc_id, text)
        doc_id = doc_id_or_corpus
        self._pending_ids.append(doc_id)
        self._pending_texts.append(text)
        self._index_built = False

    def _flush_index(self) -> None:
        """Build the MTEB index from all pending documents."""
        if not self._pending_ids:
            return
        with self._engine_lock:
            self._engine.auto_activate_mteb()
            self._engine.index_mteb(
                self._pending_ids, self._pending_texts, "retrieval", None
            )
        self._pending_ids.clear()
        self._pending_texts.clear()
        self._index_built = True

    def build_index(self) -> None:
        """Explicitly build the search index from all accumulated documents."""
        self._flush_index()

    def search(
        self,
        query_or_queries=None,
        top_k: int = 10,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        encode_kwargs=None,
        top_ranked=None,
        num_proc=None,
        **kwargs,
    ):
        """Search indexed documents. Use via mteb.evaluate() for retrieval; or search(query, top_k) for direct use."""
        # MTEB protocol: search(queries=..., top_k=..., task_metadata=..., ...) — all keyword args
        if task_metadata is not None:
            queries = query_or_queries if query_or_queries is not None else kwargs.get("queries")
            if queries is None:
                raise ValueError("search() for MTEB requires queries (positional or keyword)")
            top_k = kwargs.get("top_k", top_k)
            task = str(getattr(task_metadata, "name", ""))
            qids = list(queries["id"])
            try:
                from mteb._create_dataloaders import _create_text_queries_dataloader
                qtexts = [t for b in _create_text_queries_dataloader(queries) for t in b["text"]]
            except Exception:
                qtexts = [str(q) for q in queries.get("text", [])]
            with self._engine_lock:
                raw = self._engine.search_mteb(qids, qtexts, task, top_k, None)
            return {
                qid: {did: float(s) for did, s in docs.items()}
                for qid, docs in raw.items()
            }

        # Simple API: search(query, top_k=10)
        self._flush_index()
        qid = "_q"
        with self._engine_lock:
            raw = self._engine.search_mteb([qid], [query_or_queries], "retrieval", top_k, None)
        doc_scores = raw.get(qid, {})
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Backward-compatible aliases
    encode_doc = index
    recall = search

    # ========================================================================
    # UTILITY
    # ========================================================================

    def truncate_embeddings(
        self,
        embeddings: np.ndarray,
        target_dim: int = 256,
    ) -> np.ndarray:
        """Truncate embeddings (Matryoshka Representation Learning)."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        with self._engine_lock:
            return self._engine.truncate_embeddings(embeddings, target_dim)

    def count_tokens(self, text: str) -> int:
        """Count BERT tokens for a text."""
        with self._engine_lock:
            return self._engine.count_tokens(text)

    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID."""
        with self._engine_lock:
            return self._engine.get_document(doc_id)

    def __len__(self) -> int:
        """Number of indexed documents (including pending)."""
        return self._engine.doc_count() + len(self._pending_ids)

    def stats(self) -> dict:
        """Get model statistics."""
        return self._engine.stats()

    def __repr__(self) -> str:
        return f"LAM(tier={self.tier}, max_tokens={self.max_tokens:,}, docs={len(self)})"


__all__ = [
    "LAM",
    "TIER_FREE",
    "TIER_BETA",
    "TIER_LICENSED",
    "TIER_INFINITE",
    "__version__",
]
