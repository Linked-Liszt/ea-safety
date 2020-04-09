import gym
from evo_ac.logger import EvoACLogger
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
import sys
import json

def test_model(model):
    obs = env.reset()
    fitness = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        fitness += rewards[0]
        if dones[0]:
            break
    return fitness


config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

logger = EvoACLogger(config_dict)

exp_config = config_dict['experiment']






# Parallel environments
env = make_vec_env('CartPole-v1', n_envs=exp_config['num_envs'])

for run in range(exp_config['num_runs']):
    model = A2C(MlpPolicy, env, verbose=1)
    for epoch in range(exp_config['num_gens']):
        model.learn(total_timesteps=1000)
    
        fitness = test_model(model)
        logger.save_fitnesses({}, [fitness], 0.0, 0.0, epoch, epoch * 1000)
        logger.print_data(epoch)
    logger.end_run()

logger.end_experiment()