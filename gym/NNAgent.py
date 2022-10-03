import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import deque

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

    def act(self, state):
        if random.randrange(100) <= self.exploration_rate:
            return random.randrange(self.action_size) 
        else:
            return np.argmax(self.model.predict(state)[0])

