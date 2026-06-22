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

REPLAY_VERSION = 1
"""version of the routing-only format the rust crates replay. see
[`RouterTrace.write_replay`] for the byte layout."""

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
    segments: np.ndarray | None = None
    """token index where each corpus segment began, ascending from 0.

    a segment is a stretch of text about one thing. carrying the boundaries
    makes it possible to ask whether routing depends on subject matter, which is
    the claim the cache policy rests on and which cannot be recovered from the
    routing alone.
    """

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
        if self.segments is not None:
            self.segments = np.ascontiguousarray(self.segments, dtype=np.int64)
            if self.segments.size and (
                self.segments.min() < 0 or self.segments.max() >= max(self.n_tokens, 1)
            ):
                raise ValueError(
                    f"segment boundaries {self.segments.tolist()} fall outside "
                    f"a trace of {self.n_tokens} tokens"
                )
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

    @property
    def has_segments(self) -> bool:
        return self.segments is not None and self.segments.size > 1

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
        if self.segments is not None:
            arrays["segments"] = self.segments
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
                segments=f["segments"] if "segments" in f.files else None,
            )

    def write_replay(self, path: str | Path) -> Path:
        """write the routing only, in a form the rust crates can read.

        # why a second format exists

        the npz carries hidden states and is hundreds of megabytes, needs numpy
        to open, and the rust crates deliberately have no dependencies. the
        cache does not need any of that. it needs the access order and nothing
        else, and that is 800kb rather than 200mb, small enough to live in the
        repository so the replay is reproducible rather than described.

        # layout, little endian

        ```
        0   8  magic "STRTRACE"
        8   4  format version
        12  4  n_tokens
        16  4  n_layers
        20  4  n_experts
        24  4  top_k
        28  .. n_tokens * n_layers * top_k u16 expert indices, token major
        ```

        expert indices rather than packed keys, so the reader has to rebuild the
        expert-layer pair itself and the two sides cannot silently disagree
        about the packing. u16 caps this at 65536 experts per layer, which is
        two orders of magnitude past anything shipping.
        """
        if self.n_experts > 0xFFFF:
            raise ValueError(
                f"{self.n_experts} experts does not fit the u16 this format "
                f"uses for an expert index"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            b"STRTRACE"
            + REPLAY_VERSION.to_bytes(4, "little")
            + self.n_tokens.to_bytes(4, "little")
            + self.n_layers.to_bytes(4, "little")
            + self.n_experts.to_bytes(4, "little")
            + self.top_k.to_bytes(4, "little")
        )
        body = self.routing.astype("<u2").tobytes()
        path.write_bytes(header + body)
        return path

    def describe(self) -> str:
        hidden = f", hidden dim {self.hidden.shape[2]}" if self.hidden is not None else ""
        if self.has_segments:
            hidden += f", {self.segments.size} segments"
        return (
            f"{self.n_tokens} tokens x {self.n_layers} layers x top-{self.top_k} "
            f"of {self.n_experts} experts{hidden}\n{self.provenance.banner()}"
        )
