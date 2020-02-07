"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import safety_gym
import gym

#ENVIRONMENT = 'Safexp-PointGoal1-v0'
ENVIRONMENT = 'Ant-v2'

class TestNeat:
    _episode_counter = 0

    def env_setup(self):
        self.gym_env = gym.make(ENVIRONMENT)

    def env_teardown(self):
        self.gym_env.close()

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)    
            obs = self.gym_env.reset()
            
            #print(obs)
            #print(self.gym_env.action_space.low)
            
            

            fitness = 0
            while True:
                #print(net.activate(obs))
                #input()

                action = net.activate(obs)
                
                for i in range(len(action)):
                    action[i] = (action[i]-0.5) * 2
                #action = round(action[0])

                obs, reward, done, hazards = self.gym_env.step(action) 

                fitness += reward
                
                if done:
                    break
            
            genome.fitness = fitness

            self._episode_counter += 1
            #print(self._episode_counter)
            
    def demo_best_net(self):
        obs = self.gym_env.reset()

        while True:
                self.gym_env.render()
                action = self.winner_net.activate(obs)

                action = round(action[0])

                obs, reward, done, hazards = self.gym_env.step(action) 
                
                if done:
                    break

    def run(self, config_file):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(self.eval_genomes, 150)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        self.winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.ini')
    curr_env = TestNeat()
    curr_env.env_setup()
    curr_env.run(config_path)
    curr_env.demo_best_net()
    curr_env.env_teardown()