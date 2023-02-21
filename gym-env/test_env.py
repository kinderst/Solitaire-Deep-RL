import random
import numpy as np
import gym
import time

from solitare_env import SolitaireWorldEnv

env = SolitaireWorldEnv(render_mode="human")

for i in range(num_episodes):
	print("episode", i)
	state, info = env.reset()

	terminated = None
	truncated = None
	print("initial actions: ", info['action_mask'])

	while not terminated and not truncated:


		a = input("select action")

		state, reward, terminated, info = env.step(a)
		print("actions after step: ", info['action_mask'])
