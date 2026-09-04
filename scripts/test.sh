#!/usr/bin/env bash
# run everything. from windows, invoke through wsl:
#   wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Pawan/Desktop/strata && ./scripts/test.sh"
set -euo pipefail
cd "$(dirname "$0")/.."

# building on /mnt/c through the 9p mount is slow, so keep artefacts on the
# linux filesystem
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/strata-target}"

echo "== rust =="
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo
echo "== m0 =="
cd m0
python3 -m pytest tests -q
