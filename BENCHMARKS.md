# Reproduction and benchmark scope

A new Pendulum-v1 path trains the released actor/critic with correct action scaling and time-limit handling. Three30-episode runs retain validation-selected checkpoints, held-out returns and random/zero-action baselines. Multi-agent transition batches are validated before replay mutation.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `control-benchmark` | Configurable Gymnasium benchmark with default 200-episode budget; distinct from the shorter published reference | None |
| `reference-control` | Published three-seed bounded Gymnasium budget; held-out rollout seeds and random/zero-action baselines; no Unity solve claim | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The bounded runs underperform the random baseline. They are not a Reacher reproduction, real-robot result or control-safety validation.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

```sh
python -m pip install -r requirements.txt
python train.py --episodes 30 --seeds 0,1,2 --output runs/pendulum
python train.py --evaluate-checkpoint checkpoints/checkpoint-seed-0.pt --output runs/reloaded
```

The actor's normalized output is scaled to Pendulum's action limits only at the environment boundary. Replay keeps normalized actions, scales training reward by 0.1 and bootstraps time-limit truncations. Evaluation reports original undiscounted returns. `results/gymnasium/benchmark.json` includes all returns and weak outcomes.


## Recorded command definitions

### checks

Offline behavioral regressions; not a fresh dataset benchmark

```text
{python} smoke_test.py
{python} test_algorithm.py
```

### control-benchmark

Configurable Gymnasium benchmark with default 200-episode budget; distinct from the shorter published reference

```text
{python} train.py --output {output}/control
```

### reference-control

Published three-seed bounded Gymnasium budget; held-out rollout seeds and random/zero-action baselines; no Unity solve claim

```text
{python} train.py --episodes 30 --output {output}/reference-control
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.
