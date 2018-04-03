from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__file__.split(os.sep)[-1])
logger.setLevel(logging.INFO)


import os, gym
import time

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym_id', default='CartPole-v0', help="Name of the OpenAI Gym Environment")
    parser.add_argument('-a', '--agent', type=str, default='PPO', help="Agent to train.")
    parser.add_argument('-e', '--episodes', type=int, default=20, help="Number of episodes to train for.")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps to train for.")
    parser.add_argument('-nv', '--novisualize', action='store_false', default=True, help="Don't visualize training (will speed up training)")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose deterministically and don't use random actions.")
    parser.add_argument('-l', '--load', help="Load pretrained agent from this particular directory.")
    parser.add_argument('-x', '--exp', type=str, default='exp', help="Name of experiment for logging/saving weights.")
    parser.add_argument('--monitor', default='./logs/', help="Save results and logs to this directory.")
    parser.add_argument('--save', default='./weights/', help="Save trained model to this directory.")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results.")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled).")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs.")

    args = parser.parse_args()
    logfilepath = os.path.join(args.monitor, args.agent, args.exp)
    logger.info('Creating logging folder {}'.format(logfilepath))
    os.system('mkdir -p {}'.format(logfilepath))
    env = OpenAIGym(
        gym_id=args.gym_id,
        monitor=logfilepath,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        visualize=args.novisualize
    )

    # Load the required agent from custom module
    logger.info('Loading {} Agent/Network'.format(args.agent))

    if args.agent.lower() == 'ddpg':
        from modules.custom_agents import DDPG_Agent_Network
        agent, network = DDPG_Agent_Network()
    elif args.agent.lower() == 'naf':
        from modules.custom_agents import NAF_Agent_Network
        agent, network = NAF_Agent_Network()
    elif args.agent.lower() == 'trpo':
        from modules.custom_agents import TRPO_Agent_Network
        agent, network = TRPO_Agent_Network()
    elif args.agent.lower() == 'ppo':
        from modules.custom_agents import PPO_Agent_Network
        agent, network = PPO_Agent_Network()
    elif args.agent.lower() == 'vpg':
        from modules.custom_agents import VPG_Agent_Network
        agent, network = VPG_Agent_Network()

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=env.states,
            actions=env.actions,
            network=network,
        )
    )

    if args.load:
        logger.info("Testing pre-trained model!")
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)
        logger.info('Loaded pre-trained model weights!')
        logger.info('Starting testing process!')
        env = gym.make(args.gym_id)
        s = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.act(s)
            s, r, done, _ = env.step(action)
        return

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=env,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
    filepath = os.path.join(args.save, args.agent, args.exp)
    logger.info('Creating directory {}'.format(filepath))
    os.system('mkdir -p {}'.format(filepath)) # recursive mkdir
    logger.info("Saving trained model to {}!".format(filepath))
    filepath = agent.save_model(os.path.join(filepath, 'model'), append_timestep=False)
    logger.info("Saved trained model as: {}".format(filepath))

    runner.close()

if __name__ == '__main__':
    main()
