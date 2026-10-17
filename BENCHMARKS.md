# Reproduction and benchmark scope

A new Pendulum-v1 path trains the released actor/critic with correct action scaling and time-limit handling. Three 30-episode runs retain validation-selected checkpoints, held-out returns and random/zero-action baselines. Multi-agent transition batches are validated before replay mutation.

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

The original 30-episode runs underperform the random baseline. The 100-episode extension improves the overall mean over random, with substantial seed variation and weaker results than the task-aware controller below. They are not a Reacher reproduction, real-robot result or control-safety validation.

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

## Longer-budget diagnostic and task-aware control baseline

An additional fixed **100-episode × three-seed** run retains [all learning traces and returns](results/gymnasium-100/benchmark.json), selected checkpoints, [controller rollouts](results/gymnasium-100/controller.json) and [source revision/commands](results/gymnasium-100/protocol.json). Selection uses validation reset seeds only. Previously inspected final rollout seeds are reused: this is an exploratory budget extension, not untouched independent validation.

| Policy | Mean return | Across training-seed groups |
|---|---:|---:|
| Released RL agent, 100 episodes | -1037.88 | sample SD 313.24 |
| Random actions | -1203.13 | same reset seeds |
| Fixed task-aware controller | -665.26 | same reset seeds |

The longer DDPG run improves its mean over random, but variation is large and one training seed still underperforms random. It remains worse than the fixed energy-shaping/PD controller. This does not establish a robust Pendulum solution, a Reacher reproduction or physical safety.

The fixed controller is a **task-aware engineering reference**, with a knowledge advantage over model-free learning. CartPole uses a declared linear switching rule; Pendulum uses known simulator physics for energy shaping and a fixed local PD region. All constants were fixed before the controller's first evaluation. They were not tuned against these returns. Input/action conventions follow [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/). The complete implementation is in [benchmark_controller.py](benchmark_controller.py).

```sh
python train.py --episodes 100 --seeds 0,1,2 --output runs/extended
python benchmark_controller.py --benchmark runs/extended/benchmark.json --output runs/controller
python train.py --evaluate-checkpoint results/gymnasium-100/checkpoint-seed-0.pt --output runs/extended-reload
```

## Pinned local reference installation

[requirements-reference-macos.txt](requirements-reference-macos.txt) pins the installed runtime/test dependency closure for Python 3.14.3 on macOS ARM64. It accompanies the observed environment record. A new empty Python3.14.3 macOS environment installed these pinned dependencies, passed `pip check`, and passed all eight offline repository check profiles. [The validation record](environment-validation.json) binds the source revisions, Python source hashes, versions and outcomes. This is a platform-specific reference, not a Linux/CUDA lock or a repetition of every full dataset benchmark. The GitHub checks use their separately declared Linux environment.

```sh
python3.14 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
```

Use the recorded exact Python patch version when comparing numerical results. Notebook tooling and optional FID dependencies are outside this runtime/test snapshot.
