# Visualize the output of the PPO Agent and record the results

import os
import random
import numpy as np
import gym
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gym_env import solitaire_env

#from gym.wrappers import Monitor
from gym.wrappers.record_video import RecordVideo

mdir = save_location = os.path.dirname(os.path.realpath(__file__)) + '/recordings'

env = solitaire_env.SolitaireWorldEnv(render_mode="human")
#env = Monitor(env, directory=save_location, force=True, video_callable=lambda episode_id: True)
env = RecordVideo(env, mdir, episode_trigger=lambda e_idx:True)
env.reset()
env.start_video_recorder()

def get_suit_val(card_param):
    if card_param == 0:
        return 0
    elif card_param == 53:
        return -1
    
    if card_param / 13 <= 1:
        return 1
    elif card_param / 26 <= 1:
        return 2
    elif card_param / 39 <= 1:
        return 3
    else:
        return 4

def get_card_val(card_p):
    if card_p == 0:
        return 0
    elif card_p == 53:
        return -1
    
    num = card_p % 13
    if num == 0:
        return 13
    else:
        return num

suit_val_vec = np.vectorize(get_suit_val)
card_val_vec = np.vectorize(get_card_val)


# recreate the model arch

# define multi-layer perceptron used in model arch
def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)

# Sample action from actor
@tf.function
def sample_action(observation, impossible_actions_param):
    logits = actor(observation)
    #reshape indices to fit spec of tensor_scatter_nd_update
    impossible_action_indices = tf.reshape(impossible_actions_param, [impossible_actions_param.shape[0],-1])
    #update impossible actions to have -inf, so even the log-probability
    #of choosing them is 0
    neg_inf_updates = np.zeros(impossible_action_indices.shape[0])
    neg_inf_updates.fill(-np.inf)
    possible_logits = tf.tensor_scatter_nd_update(logits[0], impossible_action_indices, neg_inf_updates)

    possible_logits_reshaped = tf.reshape(possible_logits, [1,-1])

    # action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    action = tf.squeeze(tf.random.categorical(possible_logits_reshaped, 1), axis=1)

    return logits, action

hidden_sizes = (128, 128)
observation_dimensions = 242
num_actions = env.action_space.n
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)

actor.load_weights("ppo_actor_weights.h5")

num_episodes = 1

for i in range(num_episodes):
	print("episode", i)
	observation, info = env.reset()
	deck_num = observation["deck_position"]
	suits = observation['suits']
	decks = observation['decks'].flatten()
	decks_suits = suit_val_vec(decks)
	decks_card_vals = card_val_vec(decks)
	piles = observation['piles'].flatten()
	piles_suits = suit_val_vec(piles)
	piles_card_vals = card_val_vec(piles)
	piles_behind = (observation['piles_behind'] != 0).sum(0)
	observation = np.concatenate((np.array([deck_num]),suits,decks_suits,decks_card_vals,piles_suits,piles_card_vals,piles_behind))
	observation = observation.reshape(1, -1)
	impossible_actions = np.nonzero(info['action_mask'] == 0)[0]

	terminated = None
	truncated = None
	time.sleep(1)

	while not terminated:
		logits, action = sample_action(observation, impossible_actions)

		observation = observation.reshape(1, -1)
		observation_new, reward, terminated, info = env.step(action[0].numpy())

		deck_num = observation_new["deck_position"]
		suits = observation_new['suits']
		decks = observation_new['decks'].flatten()
		decks_suits = suit_val_vec(decks)
		decks_card_vals = card_val_vec(decks)
		piles = observation_new['piles'].flatten()
		piles_suits = suit_val_vec(piles)
		piles_card_vals = card_val_vec(piles)
		piles_behind = (observation_new['piles_behind'] != 0).sum(0)
		observation = np.concatenate((np.array([deck_num]),suits,decks_suits,decks_card_vals,piles_suits,piles_card_vals,piles_behind))
		observation = observation.reshape(1, -1)
		impossible_actions = np.nonzero(info['action_mask'] == 0)[0]

env.close()