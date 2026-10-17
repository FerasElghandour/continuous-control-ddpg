# Parallel Continuous Control with DDPG

Control continuous robotic-arm actions while learning from experience shared across parallel simulated agents.

## Implementation

Actor–critic networks, shared replay across 20 agents, Ornstein–Uhlenbeck exploration, deferred gradient updates, critic clipping, and soft target-network updates.

![Implementation mechanism: State batch from parallel agents → Local actor produces continuous actions → Shared replay stores transitions → Critic target uses target actor and critic → Update local networks and softly update targets](assets/mechanism.svg)



## Train the continuous-control agent

A new Pendulum-v1 path trains the released actor/critic with correct action scaling and time-limit handling. Three 30-episode runs retain validation-selected checkpoints, held-out returns and random/zero-action baselines. Multi-agent transition batches are validated before replay mutation.

Longer-budget evidence: **-1037.88 mean return after 100 episodes per seed**, versus random -1203.13 and a task-aware controller -665.26. The mean improves over random, with substantial seed variation; the controller remains stronger. These are exploratory simulator comparisons, not legacy environment or real-robot results.

[Commands, artifacts and benchmark scope](BENCHMARKS.md) · [Versioned profiles](reproduction.json)


## Try a small example

Generated transitions exercise actor inference and one replay update. No simulator is launched and no control-policy performance is measured.

After the local setup below, run from the repository root:

```sh
python examples/quickstart.py
```

The [example source](examples/quickstart.py), [captured output](examples/expected-output.txt) and [execution record](examples/environment.json) keep this demonstration inspectable. Floating-point values may vary with library versions.

Observed output from this run:

```text
{
  "fixture": "untrained actor and generated transitions; no rollout",
  "actions": [
    [
      -0.06248923018574715,
      0.07946766167879105
    ],
    [
      -0.06248923018574715,
      0.07946766167879105
    ],
    [
      -0.06248923018574715,
      0.07946766167879105
    ]
  ],
  "replay_size": 8,
  "actor_changed": true
}
```

## Evidence and results

The recovered implementation configures a 100-episode rolling score target of 30. The release tests action bounds, replay sampling, gradient updates, and target-network behavior. No numerical training reward is claimed without a recovered reward trace.

See [VALIDATION.md](VALIDATION.md) for the release checks and their scope.

## Run locally

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Continuous_Control.py
# Full simulation also requires a compatible legacy unityagents installation.
```

Prepare the inputs described in [DATA.md](DATA.md) first. Source validation is available without installing model dependencies:

```sh
python scripts/validate_source.py
```

Run the focused offline behavior checks with `python smoke_test.py` after installing the dependencies.

## Limitations

Legacy Unity Reacher binaries and Reacher-trained weights are not included. The separate Gymnasium Pendulum benchmark includes trained checkpoints and evaluation records. Full reproduction requires the legacy Reacher environment and a compatible unityagents runtime; the modern CPU checks exercise the learning core independently.

## Reuse

Original reproduction and validation tooling is available under MIT; see [LICENSING.md](LICENSING.md) for the explicit file scope and preserved upstream, dataset and checkpoint terms.
