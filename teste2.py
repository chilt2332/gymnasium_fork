from stable_baselines3 import PPO
from gymnasium.envs.registration import make
import os

models_dir = "models/PPO_"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = make('CarRacing-v3')
env.reset()

model = PPO('CnnPolicy', env, verbose=1)

TIMESTEPS = 10000
MAX_ITERATIONS = 10
iters = 0

while iters < MAX_ITERATIONS:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/ppo_car_racing_{TIMESTEPS * iters}")

env.close()
