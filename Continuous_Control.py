"""
Continuous_Control.py — Training script for the Reacher Continuous Control project.

Trains a DDPG agent to control a double-jointed robotic arm to reach
target locations (20-agent version).

Run from the project root directory:
    python Continuous_Control.py

The Unity environment must be downloaded first — see README.md.
"""

import os
import sys
import platform
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

from ddpg_agent import Agent

# ── 1. Locate the Unity environment ──────────────────────────────────────────
def find_env():
    """Return the Reacher environment binary path for the current platform."""
    system = platform.system()
    candidates = {
        "Darwin": [
            "Reacher_20.app",
            "Reacher.app",
        ],
        "Linux":  [
            "Reacher_Linux/Reacher.x86_64",
            "Reacher_Linux/Reacher.x86",
            "Reacher_Linux_NoVis/Reacher.x86_64",
        ],
    }
    for p in candidates.get(system, []):
        if os.path.exists(p):
            return p
    print(
        "ERROR: Reacher Unity environment not found.\n"
        "Please download it and place it in the project root.\n"
        "See README.md for download links."
    )
    sys.exit(1)

if __name__ == "__main__":
    from unityagents import UnityEnvironment
    env_path = find_env()
    print(f"Loading environment from: {env_path}")
    env = UnityEnvironment(file_name=env_path)

    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]

    # ── 2. Inspect the environment ────────────────────────────────────────────────
    env_info    = env.reset(train_mode=True)[brain_name]
    num_agents  = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size  = env_info.vector_observations.shape[1]

    print(f"\nNumber of agents  : {num_agents}")
    print(f"State  space size : {state_size}")
    print(f"Action space size : {action_size}\n")

    # ── 3. Create the agent ───────────────────────────────────────────────────────
    agent = Agent(
        state_size  = state_size,
        action_size = action_size,
        num_agents  = num_agents,
        seed        = 42,
    )

    # ── 4. Training loop ──────────────────────────────────────────────────────────

    def ddpg(
        n_episodes   = 300,
        max_t        = 1000,
        score_target = 30.0,
        score_window = 100,
        print_every  = 10,
    ):
        """DDPG training loop for the 20-agent Reacher environment.

        Params
        ======
            n_episodes   (int):   maximum number of training episodes
            max_t        (int):   maximum steps per episode
            score_target (float): rolling average score considered 'solved'
            score_window (int):   window size for the rolling average
            print_every  (int):   print progress every N episodes
        """
        scores_all    = []                          # score of every episode
        scores_window = deque(maxlen=score_window)  # rolling window

        for i_episode in range(1, n_episodes + 1):
            env_info = env.reset(train_mode=True)[brain_name]
            states   = env_info.vector_observations          # (num_agents, state_size)
            agent.reset()
            episode_scores = np.zeros(num_agents)

            for _ in range(max_t):
                actions    = agent.act(states, add_noise=True)
                env_info   = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards     = env_info.rewards
                dones       = env_info.local_done

                agent.step(states, actions, rewards, next_states, dones)

                states          = next_states
                episode_scores += rewards

                if np.any(dones):
                    break

            avg_score = np.mean(episode_scores)   # average over all 20 agents
            scores_window.append(avg_score)
            scores_all.append(avg_score)
            rolling_avg = np.mean(scores_window)

            print(
                f"\rEpisode {i_episode:4d}  "
                f"Avg Score (last {score_window}): {rolling_avg:.2f}",
                end="",
            )
            if i_episode % print_every == 0:
                print(
                    f"\rEpisode {i_episode:4d}  "
                    f"Avg Score (last {score_window}): {rolling_avg:.2f}"
                )

            if rolling_avg >= score_target and i_episode >= score_window:
                print(
                    f"\n✓ Solved in {i_episode - score_window:d} episodes!"
                    f"  Avg Score: {rolling_avg:.2f}"
                )
                torch.save(agent.actor_local.state_dict(),  "checkpoint_actor.pth")
                torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
                print("Weights saved → checkpoint_actor.pth, checkpoint_critic.pth")
                break

        return scores_all


    scores = ddpg()

    # ── 5. Plot rewards ───────────────────────────────────────────────────────────
    rolling_avg = [np.mean(scores[max(0, i - 99):i + 1]) for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(scores)), scores, alpha=0.4, color="steelblue", label="Episode score")
    ax.plot(np.arange(len(scores)), rolling_avg, color="orange", linewidth=2, label="100-ep average")
    ax.axhline(y=30.0, color="red", linestyle="--", linewidth=1.5, label="Target (+30)")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Score")
    ax.set_title("DDPG — Continuous Control (Reacher, 20 agents)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scores.png", dpi=150)
    print("Plot saved → scores.png")

    env.close()
