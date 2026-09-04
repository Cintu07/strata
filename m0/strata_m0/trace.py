"""the router trace: what m0 measures, and where every number comes from.

a trace is the routing decisions of a real moe model on a real corpus, plus
optionally the hidden states that fed each router. every plot in the m0 writeup
is a function of one of these, so the schema is the contract for the whole
milestone.

provenance is part of the schema and not an afterthought. a synthetic trace and
a captured one are the same shape, which is exactly why every trace has to say
which it is, and why the report refuses to present a synthetic one as a result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

#: a trace that was captured from a real model on a real corpus
SOURCE_CAPTURED = "captured"
#: a trace generated with a known structure, for testing the harness itself
SOURCE_SYNTHETIC = "synthetic"


@dataclass
class Provenance:
    """where a trace came from, carried with it everywhere."""

    source: str
    model_id: str
    corpus: str = ""
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def is_real(self) -> bool:
        return self.source == SOURCE_CAPTURED

    def banner(self) -> str:
        """the line that goes at the top of anything derived from this trace."""
        if self.is_real:
            return f"captured from {self.model_id} on {self.corpus or 'an unnamed corpus'}"
        return (
            f"SYNTHETIC trace, generated structure, not a measurement. "
            f"nominal model {self.model_id}. {self.notes}".strip()
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model_id": self.model_id,
            "corpus": self.corpus,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        return cls(
            source=d["source"],
            model_id=d["model_id"],
            corpus=d.get("corpus", ""),
            created_at=d.get("created_at", 0.0),
            notes=d.get("notes", ""),
        )


@dataclass
class RouterTrace:
    """routing decisions for every token, layer, and slot.

    attributes:
        routing: int32 ``[n_tokens, n_layers, top_k]`` of expert indices. the
            router's top-k choice for each token at each layer.
        n_experts: experts per layer in the model, which is not the same as the
            number the trace happened to touch.
        hidden: optional float32 ``[n_tokens, n_layers, d_probe]``, the hidden
            state entering each layer's router. this is what the speculative
            router head would have to predict from, and it is large, so it is
            usually captured at a reduced dimension.
        provenance: where this came from.
    """

    routing: np.ndarray
    n_experts: int
    provenance: Provenance
    hidden: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.routing.ndim != 3:
            raise ValueError(
                f"routing must be [n_tokens, n_layers, top_k], got shape {self.routing.shape}"
            )
        self.routing = np.ascontiguousarray(self.routing, dtype=np.int32)
        if self.routing.size and self.routing.max() >= self.n_experts:
            raise ValueError(
                f"routing references expert {self.routing.max()} "
                f"but the model has {self.n_experts}"
            )
        if self.routing.size and self.routing.min() < 0:
            raise ValueError("routing contains a negative expert index")
        if self.hidden is not None:
            if self.hidden.shape[:2] != self.routing.shape[:2]:
                raise ValueError(
                    f"hidden {self.hidden.shape[:2]} does not line up with "
                    f"routing {self.routing.shape[:2]}"
                )
            self.hidden = np.ascontiguousarray(self.hidden, dtype=np.float32)

    # ----------------------------------------------------------------- shape

    @property
    def n_tokens(self) -> int:
        return int(self.routing.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.routing.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.routing.shape[2])

    @property
    def has_hidden(self) -> bool:
        return self.hidden is not None

    def experts_at(self, token: int, layer: int) -> np.ndarray:
        """the distinct experts routed for one token at one layer."""
        return np.unique(self.routing[token, layer])

    def layer(self, layer: int) -> np.ndarray:
        """``[n_tokens, top_k]`` for one layer."""
        return self.routing[:, layer, :]

    def flat_keys(self) -> np.ndarray:
        """the whole trace as packed expert-layer keys, in access order.

        packed the same way ``strata_format::ExpertKey::packed`` does, so a
        trace written here can be replayed through the rust cache without a
        translation step that could quietly disagree.
        """
        n_tokens, n_layers, top_k = self.routing.shape
        layers = np.broadcast_to(
            np.arange(n_layers, dtype=np.int64)[None, :, None],
            (n_tokens, n_layers, top_k),
        )
        return ((layers << 32) | self.routing.astype(np.int64)).reshape(-1)

    # -------------------------------------------------------------------- io

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"routing": self.routing}
        if self.hidden is not None:
            arrays["hidden"] = self.hidden
        meta = {
            "schema_version": SCHEMA_VERSION,
            "n_experts": self.n_experts,
            "provenance": self.provenance.to_dict(),
        }
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RouterTrace":
        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(bytes(f["meta"]).decode("utf-8"))
            if meta["schema_version"] != SCHEMA_VERSION:
                raise ValueError(
                    f"trace schema version {meta['schema_version']} is not "
                    f"version {SCHEMA_VERSION} that this build reads"
                )
            return cls(
                routing=f["routing"],
                n_experts=meta["n_experts"],
                provenance=Provenance.from_dict(meta["provenance"]),
                hidden=f["hidden"] if "hidden" in f.files else None,
            )

    def describe(self) -> str:
        hidden = f", hidden dim {self.hidden.shape[2]}" if self.hidden is not None else ""
        return (
            f"{self.n_tokens} tokens x {self.n_layers} layers x top-{self.top_k} "
            f"of {self.n_experts} experts{hidden}\n{self.provenance.banner()}"
        )
