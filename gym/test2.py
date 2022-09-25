import gym #gym library
#Create environment, choose game and the render_mode for visuals
env = gym.make("LunarLander-v2", render_mode="human")

#take some actions from the space of all possible actions. (Preditermined choice by Seed 42)
env.action_space.seed(42)

observation, info = env.reset(seed=42)

#Game loop
for _ in range(1000):
    #make a random move from the space of all current possible actions
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    #if crased reset the environment
    if terminated or truncated:
        observation, info = env.reset()

#close the program
env.close()