"""Fixed task-aware controller diagnostic; no training or tuning on rollout returns.

CartPole uses a declared linear switching rule. Pendulum uses energy shaping
outside a fixed upright PD region. Physical parameters come from the simulator,
a knowledge advantage over model-free RL; this is not an equal-information contest.
"""
import argparse,hashlib,json,math
from pathlib import Path
import gymnasium as gym
import numpy as np


def action(task, observation, physics):
    if task=='CartPole-v1':
        x, velocity, angle, angular_velocity=observation
        return int(angle+.1*angular_velocity+.01*x+.02*velocity>0)
    cosine,sine,velocity=observation
    angle=math.atan2(sine,cosine)
    if abs(angle)<.25 and abs(velocity)<1.:
        torque=-10.*angle-3.*velocity
    else:
        # E=I*w^2/2 + m*g*l*cos(theta)/2, I=m*l^2/3.
        # dE/dt=u*w before the simulator's speed clamp.
        energy=physics['m']*physics['l']**2*velocity**2/6+physics['m']*physics['g']*physics['l']*cosine/2
        target=physics['m']*physics['g']*physics['l']/2
        torque=(target-energy)*velocity
    return np.array([np.clip(torque,-physics['max_torque'],physics['max_torque'])],dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output directory')
    source=json.loads(a.benchmark.read_text());task=source['task']
    if task not in ('CartPole-v1','Pendulum-v1'):raise ValueError('Unsupported task')
    rows=[]
    for record in source['seeds']:
        env=gym.make(task);physics={name:getattr(env.unwrapped,name) for name in ('m','g','l','max_torque')} if task=='Pendulum-v1' else {}
        scores=[]
        for seed in record['test_seeds']:
            obs,_=env.reset(seed=seed);score=0.
            for _ in range(env.spec.max_episode_steps):
                obs,reward,terminated,truncated,_=env.step(action(task,obs,physics));score+=reward
                if terminated or truncated:break
            scores.append(float(score))
        env.close();rows.append({'training_seed_group':record['seed'],'rollout_seeds':record['test_seeds'],'controller_returns':scores,'mean_return':float(np.mean(scores)),'rl_mean_return':record['mean_return'],'random_mean_return':record['random_mean'],'physics':physics})
    result={'task':task,'protocol':'Fixed task-aware controller on the same previously inspected rollout seeds; exploratory diagnostic, not untouched external evaluation','controller':'linear switching with coefficients [0.01,0.02,1,0.1]' if task=='CartPole-v1' else 'energy shaping gain 1; upright PD gains 10,3 at |angle|<0.25 and |velocity|<1','tuning':'Constants fixed before the first run; no fitting or search','information_advantage':'Pendulum controller uses known simulator mass, length, gravity and torque bound; RL learns without these dynamics','gymnasium':gym.__version__,'numpy':np.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_sha256':hashlib.sha256(a.benchmark.read_bytes()).hexdigest(),'groups':rows,'mean_return':float(np.mean([r['mean_return'] for r in rows])),'std_across_seed_groups':float(np.std([r['mean_return'] for r in rows],ddof=1))}
    a.output.mkdir(parents=True);(a.output/'controller.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
