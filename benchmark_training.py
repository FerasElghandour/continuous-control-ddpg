"""Train the released agent on a named Gymnasium control task.

A new reference benchmark, independent of legacy Banana/Reacher results. Validation
and final evaluation reset seeds are disjoint from training and from each other.
Time-limit truncations end rollouts but do not mark Bellman targets terminal.
"""
import argparse,copy,hashlib,json,random,sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch


def main(continuous=False):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,default=200);p.add_argument('--seeds',default='0,1,2');p.add_argument('--evaluation-episodes',type=int,default=10);p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluate-checkpoint',type=Path);a=p.parse_args()
 if a.episodes<1 or a.evaluation_episodes<1 or (a.output.exists() and any(a.output.iterdir())):raise ValueError('Positive budget and new output directory required')
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 task='Pendulum-v1' if continuous else 'CartPole-v1';rows=[]
 if continuous:from ddpg_agent import Agent
 else:from dqn_agent import Agent
 def policy_state(agent):
  return copy.deepcopy({'actor':agent.actor_local.state_dict(),'critic':agent.critic_local.state_dict()} if continuous else {'q':agent.qnetwork_local.state_dict()})
 def load(agent,state):
  if continuous:agent.actor_local.load_state_dict(state['actor']);agent.critic_local.load_state_dict(state['critic'])
  else:agent.qnetwork_local.load_state_dict(state['q'])
 def evaluate(agent,seeds,mode='policy'):
  env=gym.make(task);scores=[]
  for seed in seeds:
   obs,_=env.reset(seed=seed);env.action_space.seed(seed);score=0.
   for _ in range(env.spec.max_episode_steps):
    if mode=='random':action=env.action_space.sample()
    elif mode=='zero':action=np.zeros(env.action_space.shape,dtype=np.float32) if continuous else 0
    elif continuous:action=agent.act(np.asarray([obs],dtype=np.float32),add_noise=False)[0]*env.action_space.high
    else:action=agent.act(np.asarray(obs,dtype=np.float32),eps=0.)
    obs,reward,term,trunc,_=env.step(action);score+=reward
    if term or trunc:break
   scores.append(float(score))
  env.close();return scores
 evaluation_source=torch.load(a.evaluate_checkpoint,map_location='cpu',weights_only=True) if a.evaluate_checkpoint else None
 seeds=[int(evaluation_source['seed'])] if evaluation_source else list(map(int,a.seeds.split(',')))
 for seed in seeds:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  env=gym.make(task);env.action_space.seed(seed);agent=Agent(3,1,1,seed) if continuous else Agent(4,2,seed)
  validation_seeds=[1_000_000+100*seed+i for i in range(5)]
  test_seeds=[2_000_000+100*seed+i for i in range(a.evaluation_episodes)]
  trace=[];best=-float('inf');selected=None;best_ep=None;steps=0
  if a.evaluate_checkpoint:
   ck=torch.load(a.evaluate_checkpoint,map_location='cpu',weights_only=True)
   if ck['task']!=task:raise ValueError('Wrong checkpoint task')
   load(agent,ck['model_state']);best_ep=ck['selected_episode'];selected=ck['model_state']
  else:
   for ep in range(1,a.episodes+1):
    obs,_=env.reset(seed=seed*10000+ep);score=0.
    if continuous:agent.noise.reset()
    for _ in range(env.spec.max_episode_steps):
     if continuous:
      normalized=np.random.uniform(-1,1,1).astype(np.float32) if steps<1000 else agent.act(np.asarray([obs],dtype=np.float32),add_noise=True)[0]
      action=normalized*env.action_space.high
     else:action=agent.act(np.asarray(obs,dtype=np.float32),eps=max(.05,.995**ep))
     nxt,reward,term,trunc,_=env.step(action)
     if continuous:agent.step(np.asarray([obs],dtype=np.float32),np.asarray([normalized],dtype=np.float32),np.array([reward*.1]),np.asarray([nxt],dtype=np.float32),np.array([term]))
     else:agent.step(np.asarray(obs,dtype=np.float32),action,reward,np.asarray(nxt,dtype=np.float32),term)
     steps+=1;score+=reward;obs=nxt
     if term or trunc:break
    record={'episode':ep,'return':float(score),'steps_so_far':steps}
    if ep%10==0 or ep==a.episodes:
     # Preserve training exploration RNG across validation rollouts.
     py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state()
     val=evaluate(agent,validation_seeds);random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch_state)
     record['validation_mean']=float(np.mean(val))
     if record['validation_mean']>best:best=record['validation_mean'];selected=policy_state(agent);best_ep=ep
    trace.append(record)
   load(agent,selected)
  policy=evaluate(agent,test_seeds);random_scores=evaluate(agent,test_seeds,'random');zero=evaluate(agent,test_seeds,'zero')
  ck={'task':task,'model_state':selected,'seed':seed,'selected_episode':best_ep,'training_episodes':evaluation_source['training_episodes'] if evaluation_source else a.episodes,'gymnasium':gym.__version__,'torch':str(torch.__version__)}
  path=a.output/f'checkpoint-seed-{seed}.pt';torch.save(ck,path)
  # Independent reload must reproduce fixed evaluation returns exactly.
  load(agent,torch.load(path,map_location='cpu',weights_only=True)['model_state']);assert evaluate(agent,test_seeds)==policy
  result={'seed':seed,'task':task,'training_episodes':evaluation_source['training_episodes'] if evaluation_source else a.episodes,'new_training_episodes':0 if evaluation_source else a.episodes,'evaluation_only':bool(evaluation_source),'selected_episode':best_ep,'validation_seeds':validation_seeds,'test_seeds':test_seeds,'policy_returns':policy,'random_returns':random_scores,'zero_returns':zero,'mean_return':float(np.mean(policy)),'random_mean':float(np.mean(random_scores)),'zero_mean':float(np.mean(zero)),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trace':trace}
  rows.append(result);env.close();print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
 (a.output/'benchmark.json').write_text(json.dumps({'protocol':'New Gymnasium task; validation-selected checkpoint; disjoint final reset seeds; random and zero baselines; no legacy solve claim','task':task,'gymnasium':gym.__version__,'torch':str(torch.__version__),'python':sys.version.split()[0],'continuous_reward_training_scale':.1 if continuous else 1.,'truncations_bootstrap':True,'seeds':rows,'across_seed_mean':float(np.mean([r['mean_return'] for r in rows])),'across_seed_std':float(np.std([r['mean_return'] for r in rows],ddof=1)) if len(rows)>1 else None},indent=2)+'\n')
