import gym
import os

import mujoco_py

LR = 1e-3
env = gym.make('Hopper-v2')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(50):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(2000):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            print(reward)
            if done:
                break

some_random_games_first()
