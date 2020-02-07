import safety_gym
import gym

env = gym.make('Safexp-PointGoal1-v0')

obs = env.reset()

count = 0

fit = 0



while True:
    #env.render()
    obs, reward, done, hazards = env.step(env.action_space.sample()) # take a random action
    #print(obs)
    print(len(obs))
    print(reward)
    count += 1

    fit += reward
    if done:
        break

print(count)
print(reward)
print(env.action_space.sample())
env.close()