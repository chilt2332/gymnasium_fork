from stable_baselines3 import PPO
from gymnasium.envs.registration import make
import os

models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = make('CarRacing-v3', render_mode='human')
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

best_model_path = f"{models_dir}/ppo_car_racing_{TIMESTEPS * iters}"
trained_model = PPO.load(best_model_path)

env = make('CarRacing-v3', render_mode='human')
obs, _ = env.reset()

done = False
total_reward = 0

while not done:
    action, _ = trained_model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

env.close()

print(f"Showcase completed with total reward: {total_reward}")
