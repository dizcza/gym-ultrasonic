import gym
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.callbacks import TensorBoard

import gym_ultrasonic

ENV_NAME = 'UltrasonicServo-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# env = wrappers.Monitor(env, "./tmp/gym-results", force=True)
nb_actions = env.action_space.shape[0]

observation_space_input_shape = (1,) + env.observation_space.shape

# model with actor and critic
actor = Sequential()
actor.add(Flatten(input_shape=observation_space_input_shape))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=observation_space_input_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# create dddpg agent
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(
    size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.8, target_model_update=1e-3)
agent.compile(Adam(lr=.001), metrics=['mse'])

tensorboard = TensorBoard(log_dir="logs")

agent.fit(env, nb_steps=3000, visualize=1, verbose=2, nb_max_episode_steps=1000, callbacks=[])

# save the weights
# agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# test
# agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=1000)
