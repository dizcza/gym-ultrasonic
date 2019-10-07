import gym
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Input, InputLayer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import gym_ultrasonic

ENV_NAME = 'UltrasonicServo-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# env = wrappers.Monitor(env, "./tmp/gym-results", force=True)
nb_actions = env.action_space.shape[0]

# see issue https://github.com/keras-rl/keras-rl/issues/160
observation_space_input_shape = (1,) + env.observation_space.shape

# model with actor and critic
actor = Sequential(name="actor")
actor.add(InputLayer(input_shape=observation_space_input_shape, name="observation_input"))
actor.add(Flatten(input_shape=observation_space_input_shape))
actor.add(Dense(8, activation='relu'))
actor.add(Dense(8, activation='relu'))
actor.add(Dense(8, activation='relu'))
actor.add(Dense(nb_actions, activation='tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=observation_space_input_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(1, activation='linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x, name='critic')
print(critic.summary())

# create dddpg agent
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(
    size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.8, target_model_update=1e-3)
agent.compile(Adam(lr=.001), metrics=['mse'])
tensorboard = TensorBoard(log_dir="logs", write_grads=False, write_graph=False)

agent.fit(env, nb_steps=30000, visualize=1, verbose=2, nb_max_episode_steps=5000, callbacks=[])

# save the weights
# agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# test
# agent.test(env, nb_episodes=100, visualize=True, nb_max_episode_steps=5000)
