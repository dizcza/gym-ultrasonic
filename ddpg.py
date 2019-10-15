import random
from pathlib import Path

import gym.wrappers
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Flatten, Input, InputLayer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import gym_ultrasonic

random.seed(27)
np.random.seed(27)

ENV_NAME = 'UltrasonicServo-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.seed(27)
# env = gym.wrappers.Monitor(env, "capture", force=True)
nb_actions = env.action_space.shape[0]

# see issue https://github.com/keras-rl/keras-rl/issues/160
observation_space_input_shape = (1,) + env.observation_space.shape
action_input = Input(shape=(nb_actions,), name='action_input')


def create_actor():
    actor = Sequential(name="actor")
    actor.add(InputLayer(input_shape=observation_space_input_shape, name="observation_input"))
    actor.add(Flatten(input_shape=observation_space_input_shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(8, activation='relu'))
    actor.add(Dense(nb_actions, activation='tanh'))
    return actor


def create_critic():
    observation_input = Input(shape=observation_space_input_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x, name='critic')
    return critic


def create_actor_critic_agent():
    # create ddpg agent
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.5, mu=0., sigma=.1)
    agent = DDPGAgent(nb_actions=nb_actions, actor=create_actor(), critic=create_critic(),
                      critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100,
                      nb_steps_warmup_actor=100, random_process=random_process, gamma=.8, target_model_update=1e-3)
    agent.compile(Adam(lr=.001), metrics=['mse'])
    return agent


agent = create_actor_critic_agent()
WEIGHTS_PATH = Path('weights') / 'ddpg_{}_weights.h5'.format(ENV_NAME)
WEIGHTS_PATH.parent.mkdir(exist_ok=True)
WEIGHTS_PATH = str(WEIGHTS_PATH)
tensorboard = TensorBoard(log_dir="logs", write_grads=False, write_graph=False)
checkpoint = ModelCheckpoint(WEIGHTS_PATH, save_weights_only=True, period=10)

agent.load_weights(WEIGHTS_PATH)
agent.fit(env, nb_steps=50000, visualize=0, verbose=2, nb_max_episode_steps=1000, callbacks=[tensorboard, checkpoint])

# save the weights
agent.save_weights(WEIGHTS_PATH, overwrite=True)

# test
agent.test(env, nb_episodes=3, visualize=True, nb_max_episode_steps=5000)
env.close()
