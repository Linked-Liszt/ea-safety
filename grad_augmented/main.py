import numpy as np
import gym
from ac_evo import Agent
import matplotlib.pyplot as plt
from gym import wrappers


if __name__ == '__main__':
    agent = Agent(alpha=0.01, beta=0.0005, input_dims=[4], gamma=0.99,
                  n_actions=2, layer1_size=32, layer2_size=32)

    env = gym.make('CartPole-v1')
    score_history = []
    score = 0
    num_episodes = 2500
    for i in range(num_episodes):
        print('episode: ', i,'score: %.3f' % score)


        #env = wrappers.Monitor(env, "tmp/cartpole-untrained",
        #                            video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset()
        obs_buffer = []
        reward_buffer = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            obs_buffer.append(observation_)
            reward_buffer.append(reward)
            score += reward
        agent.learn(obs_buffer, reward_buffer)
        score_history.append(score)
