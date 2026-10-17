import numpy as np
import torch
from ddpg_agent import Agent,ReplayBuffer,OUNoise,device
torch.set_num_threads(2)
agent=Agent(8,2,3,7)
agent.actor_local.eval()
actions=agent.act(np.zeros((3,8),dtype=np.float32),add_noise=False)
assert actions.shape==(3,2) and (np.abs(actions)<=1).all()
assert not agent.actor_local.training
try:agent.act(np.full((3,8),np.nan))
except ValueError:pass
else:raise AssertionError('Nonfinite observations accepted')
replay=ReplayBuffer(2,128,8,7)
state=np.zeros(8)
for i in range(8):replay.add(state,np.zeros(2),1.,state,False)
state[:]=999
sample=replay.sample()
assert sample[0].max()==0
before=[p.detach().clone() for p in agent.actor_local.parameters()]
agent._learn(sample,.99)
assert any(not torch.equal(a,b) for a,b in zip(before,agent.actor_local.parameters()))
assert all(p.grad is None for p in agent.actor_target.parameters())
a,b=OUNoise((3,2),8),OUNoise((3,2),8)
assert np.array_equal(a.sample(),b.sample())
print('DDPG: action bounds, replay ownership, finite learning, detached targets, and seeded exploration passed.')

# Invalid multi-agent shapes must fail atomically before adding any transition.
before_size=len(agent.memory)
try:agent.step(np.zeros((3,8)),np.zeros((2,2)),np.zeros(3),np.zeros((3,8)),np.zeros(3))
except ValueError:pass
else:raise AssertionError('Mismatched transition batch accepted')
assert len(agent.memory)==before_size
