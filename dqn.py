import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gym_robot
import signal
import sys

# Only allocate memory as needed
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

EPISODES = 10000
SAVE_EP = 100
AVG_REW = 25

ENV_NAME = 'AutonomousRobot-v0'


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("./save/v3.h5")
    except IOError:
        pass
    done = False

    def soft_exit(signal, frame):
        print("Abort Training \n")
        var = input("Want to save the model? y/n")
        if var == 'y':
            agent.save("./save/v3.h5")
        sys.exit(0)
    signal.signal(signal.SIGINT, soft_exit)
    reward_history = []
    avg_history = []
    time_history = []
    batch_size = 32
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        sum_reward = 0

        #print(str(e) + "/" + str(EPISODES))
        for time in range(1000):

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            if(e % 50 == 0):
                env.render()
                #print(str(reward) + " | " + str(next_state))
            sum_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(("episode: {}/{}, score: {}, time: {} e: {:.2}"
                      .format(e, EPISODES, sum_reward, time, agent.epsilon)))
                if(e % AVG_REW == 0 and e != 0):
                    avg = np.average(reward_history[e - AVG_REW:e])
                    avg_history.append(avg)

                reward_history.append(sum_reward)
                time_history.append(time)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if(e % SAVE_EP == 0 and e != 0):
            name = "./save/v3" + str(e) + ".h5"
            agent.save(name)
            plt.plot(list(range(0, e, AVG_REW)), avg_history[0:e / AVG_REW])
            plt.legend(['Average Reward'], loc='upper left')
            plt.savefig("diagrams/" + str(e) + "reward.pdf")
            plt.clf()
            plt.plot(range(SAVE_EP), reward_history[e - SAVE_EP:e])
            plt.plot(range(SAVE_EP), time_history[e - SAVE_EP:e])
            plt.legend(['Reward', 'Time'], loc='upper left')
            plt.savefig("diagrams/" + str(e) + "time.pdf")
            plt.clf()
            mv_avg = 0

    # Endresultat
    plt.plot(range(EPISODES), reward_history)
    plt.plot(range(0, EPISODES, AVG_REW), avg_history)
    plt.legend(['Reward', 'Average Reward'], loc='upper left')
    plt.savefig("diagrams/reward.pdf")
    plt.clf()
    plt.plot(range(EPISODES), reward_history)
    plt.plot(range(EPISODES), time_history)
    plt.legend(['Reward', 'Time'], loc='upper left')
    plt.savefig("diagrams/time.pdf")
