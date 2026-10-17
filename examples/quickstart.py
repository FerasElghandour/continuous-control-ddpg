"""Offline example using the public implementation; see README for evidence scope."""
import sys,json,pathlib,os
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'));os.chdir(repo)
import numpy as np,torch
from ddpg_agent import Agent,ReplayBuffer
torch.set_num_threads(2)
agent=Agent(8,2,3,7)
actions=agent.act(np.zeros((3,8),dtype=np.float32),add_noise=False)
replay=ReplayBuffer(2,128,8,7)
for i in range(8): replay.add(np.zeros(8),np.zeros(2),1.,np.zeros(8),False)
before=[p.detach().clone() for p in agent.actor_local.parameters()]
agent._learn(replay.sample(),.99)
print(json.dumps({"fixture":"untrained actor and generated transitions; no rollout","actions":actions.tolist(),"replay_size":len(replay),"actor_changed":any(not torch.equal(a,b) for a,b in zip(before,agent.actor_local.parameters()))}))
