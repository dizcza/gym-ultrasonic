import random
from pathlib import Path
import shutil

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

from gym_ultrasonic.env_logger import ExpandLogger, DataDumpLogger
from gym_ultrasonic.envs.processor import NormalizeNonNegative
from gym_ultrasonic.envs.constants import WHEEL_VELOCITY_MAX

random.seed(27)
np.random.seed(27)

ENV_NAME = 'UltrasonicServo-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.seed(27)
# env = gym.wrappers.Monitor(env, "capture", force=True)
nb_actions = env.action_space.shape[0]
print(env)

# see issue https://github.com/keras-rl/keras-rl/issues/160
observation_space_input_shape = (1,) + env.observation_space.shape
action_input = Input(shape=(nb_actions,), name='action_input')


def create_actor():
    actor = Sequential(name="actor")
    actor.add(InputLayer(input_shape=observation_space_input_shape, name="observation_input"))
    actor.add(Flatten(input_shape=observation_space_input_shape))
    actor.add(Dense(64, activation='relu'))
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
    processor = NormalizeNonNegative(sensor_max_dist=env.robot.sensor_max_dist,
                                     angle_range=env.robot.servo.angle_range,
                                     action_scale=WHEEL_VELOCITY_MAX)
    agent = DDPGAgent(nb_actions=nb_actions, actor=create_actor(), critic=create_critic(),
                      critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100,
                      nb_steps_warmup_actor=100, random_process=random_process, gamma=.8, target_model_update=1e-3,
                      processor=processor)
    agent.compile(Adam(lr=.001), metrics=['mse'])
    return agent


DATA_PATH = Path('data') / f'{ENV_NAME}.csv'
DATA_PATH.parent.mkdir(exist_ok=True)
WEIGHTS_PATH = Path('weights') / f'ddpg_{ENV_NAME}.h5'
WEIGHTS_PATH.parent.mkdir(exist_ok=True)

agent = create_actor_critic_agent()
agent.load_weights(WEIGHTS_PATH)
# agent.actor.save('weights/actor.h5')

expand_logger = ExpandLogger()
dump_logger = DataDumpLogger(fpath=DATA_PATH)
tensorboard = TensorBoard(log_dir="logs", write_grads=False, write_graph=False)
checkpoint = ModelCheckpoint(str(WEIGHTS_PATH), save_weights_only=True, period=10)
shutil.rmtree(tensorboard.log_dir, ignore_errors=True)

agent.fit(env, nb_steps=50000, visualize=1, verbose=0, nb_max_episode_steps=1000,
          callbacks=[expand_logger, tensorboard, checkpoint])

# save the weights
agent.save_weights(WEIGHTS_PATH, overwrite=True)

# test
agent.test(env, nb_episodes=5, visualize=1, nb_max_episode_steps=5000, callbacks=[dump_logger])
env.close()
