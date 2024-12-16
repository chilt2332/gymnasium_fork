from stable_baselines3 import PPO
from gymnasium.envs.registration import make

models_dir = "models/PPO_Cnn_Obstacles"

env = make("models/PPO_Cnn_Obstacles",render_mode = "human")  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/10000000.zip"
model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, trunc, done, info = env.step(action)
        env.render()
        print(ep, rewards, done)
        print("---------------")