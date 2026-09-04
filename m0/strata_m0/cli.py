"""command line entry point for the m0 harness.

```
python -m strata_m0 synth   --out traces/dev.npz
python -m strata_m0 capture --model <hf-id> --out traces/real.npz --hidden
python -m strata_m0 analyse --trace traces/real.npz --out results/real
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .trace import RouterTrace


def _synth(args: argparse.Namespace) -> int:
    from .synthetic import make_trace

    trace = make_trace(
        n_tokens=args.tokens,
        n_layers=args.layers,
        n_experts=args.experts,
        top_k=args.top_k,
        persistence=args.persistence,
        signal=args.signal,
        seed=args.seed,
        with_hidden=not args.no_hidden,
    )
    path = trace.save(args.out)
    print(trace.describe())
    print(f"wrote {path}")
    return 0


def _capture(args: argparse.Namespace) -> int:
    from .capture import CaptureConfig, capture_from_texts

    if args.corpus:
        texts = Path(args.corpus).read_text(encoding="utf-8").split("\n\n")
    else:
        print(
            "no --corpus given. a trace is only as representative as the text it "
            "was captured on, so this needs real prompts from the workload you "
            "care about, not lorem ipsum.",
            file=sys.stderr,
        )
        return 2

    config = CaptureConfig(
        model_id=args.model,
        corpus=args.corpus,
        max_tokens=args.tokens,
        top_k=args.top_k,
        probe_dim=args.probe_dim if args.hidden else 0,
        device=args.device,
        dtype=args.dtype,
    )
    trace = capture_from_texts(config, [t for t in texts if t.strip()])
    path = trace.save(args.out)
    print(trace.describe())
    print(f"wrote {path}")
    return 0


def _analyse(args: argparse.Namespace) -> int:
    from .report import run

    trace = RouterTrace.load(args.trace)
    print(trace.describe())
    report = run(trace, args.out, target_fraction=args.target_fraction)
    print(f"wrote {report}")
    if not trace.provenance.is_real:
        print(
            "\nreminder: this trace is synthetic. the report says so at the top, "
            "and none of it is a result.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata_m0",
        description="measure whether the structure strata depends on actually exists",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a synthetic trace for testing the harness")
    s.add_argument("--out", default="traces/synthetic.npz")
    s.add_argument("--tokens", type=int, default=2000)
    s.add_argument("--layers", type=int, default=12)
    s.add_argument("--experts", type=int, default=32)
    s.add_argument("--top-k", type=int, default=4)
    s.add_argument("--persistence", type=float, default=0.3)
    s.add_argument("--signal", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-hidden", action="store_true")
    s.set_defaults(func=_synth)

    c = sub.add_parser("capture", help="capture a trace from a real hugging face moe model")
    c.add_argument("--model", required=True)
    c.add_argument("--corpus", required=True, help="text file, paragraphs separated by blank lines")
    c.add_argument("--out", default="traces/captured.npz")
    c.add_argument("--tokens", type=int, default=4096)
    c.add_argument("--top-k", type=int, default=None)
    c.add_argument("--hidden", action="store_true", help="record hidden states for the probe")
    c.add_argument("--probe-dim", type=int, default=128)
    c.add_argument("--device", default="cpu")
    c.add_argument("--dtype", default="float32")
    c.set_defaults(func=_capture)

    a = sub.add_parser("analyse", help="run every analysis and write the report")
    a.add_argument("--trace", required=True)
    a.add_argument("--out", default="results")
    a.add_argument("--target-fraction", type=float, default=0.2)
    a.set_defaults(func=_analyse)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
