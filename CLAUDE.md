# working in this repo

read [docs/PROGRESS.md](docs/PROGRESS.md) first. it is the state of the work in one page.

## what this project is

an inference engine for moe models that do not fit in ram, targeting a 16GB laptop running a 200B+ model. the positioning is narrow on purpose and [docs/PRD.md](docs/PRD.md) section 3 explains what is already taken. do not widen it.

## build and test

there is no msvc linker on this machine, so **cargo cannot link on windows**. run it through wsl:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Pawan/Desktop/strata && ./scripts/test.sh"
```

set `CARGO_TARGET_DIR` to a linux path. building into `/mnt/c` over the 9p mount is several times slower.

python runs natively on windows: `cd m0 && python -m pytest tests -q`.

## house rules

**measure before asserting.** thresholds in the test suites were read off `cargo test -p strata-cache --test measure -- --nocapture`, not guessed. if you add a claim, add the measurement that produced its number first, and keep the measurement in the repo.

**report what the numbers say, including when they are bad.** the cache loses to lru on one workload and there is a test that says so. the adaptive window was measured and switched off. that is the standard: a comparison that only shows the workloads it wins is not a measurement.

**never present synthetic data as a result.** every m0 trace carries provenance, and `report.py` puts a refusal banner at the top of any report built from a synthetic one. do not weaken that.

**the three crates have no external dependencies.** that is deliberate: the file format is a stable on-disk contract that other processes read, so the byte layout should be visible in the source rather than implied by a derive, and the whole workspace builds and tests offline. do not add a dependency without a reason that survives being written down in `docs/decisions/`.

**the unit is the expert-layer pair.** `ExpertKey`, never a bare expert index. expert 5 in layer 3 and expert 5 in layer 30 are unrelated tensors, and merging them makes every statistic downstream meaningless.

**do not start the engine before m0 answers.** m0 is the falsification test, not a warm-up. if the numbers say stop, the deliverable is a negative-result writeup.

## writing

lowercase plain prose. no em dashes. comments explain why a thing is the way it is, not what the line does. when a design decision has a failure mode behind it, write the failure mode down: the deadlock story in `sketch.rs` is worth more than a description of a count-min sketch.
