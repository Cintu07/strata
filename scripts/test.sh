#!/usr/bin/env bash
# run everything.
#
# on this machine there is no msvc linker, so cargo cannot link on windows and
# has to go through wsl:
#
#   wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Pawan/Desktop/strata && ./scripts/test.sh"
#
# python is the other way round: the m0 harness has numpy, matplotlib and pytest
# installed on the windows host and not inside wsl, so if this script is run
# from wsl it will tell you to run the python half natively rather than failing
# with a confusing import error.
set -euo pipefail
cd "$(dirname "$0")/.."

# building into /mnt/c over the 9p mount is several times slower, so keep
# artefacts on the linux filesystem
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/strata-target}"

echo "== rust =="
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo
echo "== m0 =="
if python3 -c "import pytest" 2>/dev/null; then
    (cd m0 && python3 -m pytest tests -q)
else
    echo "pytest is not available to $(command -v python3)."
    echo "the m0 dependencies live on the windows host, so run there instead:"
    echo
    echo "    cd m0 && python -m pytest tests -q"
    echo
    echo "or install them here with: pip install -e 'm0[dev]'"
fi
