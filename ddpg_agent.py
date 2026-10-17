import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

# ── Hyperparameters ────────────────────────────────────────────────────────────
BUFFER_SIZE   = int(1e6)  # replay buffer capacity
BATCH_SIZE    = 128       # minibatch size
GAMMA         = 0.99      # discount factor
TAU           = 1e-3      # soft-update factor for target networks
LR_ACTOR      = 1e-3      # learning rate for the actor
LR_CRITIC     = 1e-3      # learning rate for the critic
WEIGHT_DECAY  = 0         # L2 weight decay for critic optimizer
UPDATE_EVERY  = 20        # how many steps between each learning phase
NUM_UPDATES   = 10        # how many gradient updates per learning phase
GRAD_CLIP     = 1.0       # max gradient norm for critic (stability key)
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """DDPG Agent adapted for a pool of parallel agents sharing one replay buffer.

    Key stability fixes vs. a naive multi-agent extension:
      1. Gradient clipping on the critic (prevents exploding gradients).
      2. Deferred updates: learn NUM_UPDATES times every UPDATE_EVERY steps
         instead of learning at every step.
      3. Shared replay buffer — all 20 agents push experience into one buffer.
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize an Agent.

        Params
        ======
            state_size  (int): dimension of each state
            action_size (int): dimension of each action
            num_agents  (int): number of parallel agents
            seed        (int): random seed
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.num_agents  = num_agents
        self.seed        = seed
        self.t_step      = 0          # global step counter

        # Actor networks (local + target)
        self.actor_local  = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic networks (local + target)
        self.critic_local  = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Initialise target networks to match local networks
        self._hard_update(self.actor_local, self.actor_target)
        self._hard_update(self.critic_local, self.critic_target)

        # OU Noise (one process per agent)
        self.noise  = OUNoise((num_agents, action_size), seed)

        # Shared replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # ── Interaction ────────────────────────────────────────────────────────────

    def step(self, states, actions, rewards, next_states, dones):
        """Store experience from all agents; learn on a fixed schedule.

        Params
        ======
            states      (np.array): (num_agents, state_size)
            actions     (np.array): (num_agents, action_size)
            rewards     (list):     reward per agent
            next_states (np.array): (num_agents, state_size)
            dones       (list):     done flag per agent
        """
        states, actions, rewards, next_states, dones = map(np.asarray, (states, actions, rewards, next_states, dones))
        if states.shape != (self.num_agents,self.state_size) or next_states.shape != states.shape or actions.shape != (self.num_agents,self.action_size) or rewards.shape != (self.num_agents,) or dones.shape != (self.num_agents,):
            raise ValueError('Aligned state/action/reward/terminal batches required')
        if not all(np.isfinite(x).all() for x in (states,actions,rewards,next_states,dones)) or not np.isin(dones,[0,1]).all() or (np.abs(actions)>1).any():
            raise ValueError('Finite normalized transitions and binary terminal flags required')
        # All agents add to the shared buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, ns, d)

        self.t_step += 1

        # Learn every UPDATE_EVERY steps if enough samples exist
        if self.t_step % UPDATE_EVERY == 0 and len(self.memory) >= BATCH_SIZE:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self._learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Return actions for a batch of states (one per agent).

        Params
        ======
            states    (np.array): (num_agents, state_size)
            add_noise (bool):     whether to add OU exploration noise
        """
        states = np.asarray(states, dtype=np.float32)
        if states.shape != (self.num_agents, self.state_size) or not np.isfinite(states).all():
            raise ValueError("states must be a finite (num_agents, state_size) matrix")
        states = torch.from_numpy(states).float().to(device)

        was_training = self.actor_local.training
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train(was_training)

        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    # ── Learning ───────────────────────────────────────────────────────────────

    def _learn(self, experiences, gamma):
        """Update actor and critic networks using a sampled minibatch.

        Params
        ======
            experiences (Tuple[torch.Tensor]): (s, a, r, s', done) minibatch
            gamma                    (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ── Critic update ─────────────────────────────────────────────────────
        # Compute Q targets for next states using target networks
        self.actor_target.eval()
        self.critic_target.eval()
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute current Q estimates
        Q_expected = self.critic_local(states, actions)

        # MSE loss + gradient clipping for stability
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIP)
        self.critic_optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────────
        # Maximise expected return: minimise -Q(s, μ(s))
        actions_pred = self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Soft-update target networks ───────────────────────────────────────
        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_local,  self.actor_target,  TAU)

    # ── Utility ────────────────────────────────────────────────────────────────

    def _soft_update(self, local_model, target_model, tau):
        """θ_target ← τ·θ_local + (1-τ)·θ_target."""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

        for target_buffer, local_buffer in zip(target_model.buffers(), local_model.buffers()):
            if target_buffer.is_floating_point():
                target_buffer.mul_(1.0 - tau).add_(local_buffer, alpha=tau)
            else:
                target_buffer.copy_(local_buffer)

    def _hard_update(self, local_model, target_model):
        """Copy weights from local → target (initialisation only)."""
        target_model.load_state_dict(local_model.state_dict(), strict=True)


# ── Ornstein-Uhlenbeck Noise ───────────────────────────────────────────────────

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.size  = size
        self.seed  = seed
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.rng.standard_normal(self.size)
        self.state = x + dx
        return self.state


# ── Replay Buffer ──────────────────────────────────────────────────────────────

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-size replay buffer shared across all agents."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory      = deque(maxlen=buffer_size)
        self.batch_size  = batch_size
        self.seed        = seed
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(np.array(state, copy=True), np.array(action, copy=True), reward, np.array(next_state, copy=True), done))

    def sample(self):
        experiences = self.rng.sample(list(self.memory), k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
