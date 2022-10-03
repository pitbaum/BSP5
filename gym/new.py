import gym
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

env = gym.make("CartPole-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1000
output_dir = "model_output/cartpole/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_bound = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential() 
        model.add(Dense(32, activation="relu", input_dim=self.observation_space))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done 
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.exploration_rate > self.exploration_bound:
            self.exploration_rate *= self.exploration_decay

    def act(self, state):
        if random.randrange(100) <= self.exploration_rate:
            return random.randrange(self.action_space) 
        else:
            return np.argmax(self.model.predict(state)[0])

    def save(self, name): 
        self.model.save_weights(name)


for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

done = False 
time = 0
while not done:
    #env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [1, state_size]) 
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, n_episodes-1, time, agent.epsilon))
    time += 1
if len(agent.memory) > batch_size:
    agent.train(batch_size) 
if e % 50 == 0:
    agent.save(output_dir + "weights_"
               + "{:04d}".format(e) + ".hdf5")