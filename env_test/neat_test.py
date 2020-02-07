"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import safety_gym
import gym

#ENVIRONMENT = 'Safexp-PointGoal1-v0'
ENVIRONMENT = 'CartPole-v1'

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
            #print(self.gym_env.action_space.sample())

            fitness = 0
            while True:
                #print(net.activate(obs))
                #input()

                action = net.activate(obs)

                action = round(action[0])

                obs, reward, done, hazards = self.gym_env.step(action) 

                fitness += reward
                
                if done:
                    break
            
            genome.fitness = fitness

            self._episode_counter += 1
            #print(self._episode_counter)
            
    def demo_best_game(self):
        pass

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
        winner = p.run(self.eval_genomes, 300)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        p.run(self.eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.ini')
    curr_env = TestNeat()
    curr_env.env_setup()
    curr_env.run(config_path)
    curr_env.env_teardown()