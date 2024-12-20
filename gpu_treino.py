from stable_baselines3 import PPO
from gymnasium.envs.registration import make
import os
import torch  # To check GPU availability

# Check if GPU is available
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("CUDA is not available. Using CPU.")
    device = "cpu"

models_dir = "models/PPO_CNN"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print("Directory created")
else:
    print("Directory already exists")

# Initialize the environment
# env = make('CarRacing-v3', render_mode='human')
env = make('CarRacing-v3')
env.reset()

# Pass the 'device' parameter to use GPU if available
model = PPO('CnnPolicy', env, verbose=1, device=device)

TIMESTEPS = 10000
MAX_ITERATIONS = 10
iters = 0

while iters < MAX_ITERATIONS:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/ppo_car_racing_{TIMESTEPS * iters}")
    print(f"Iteration {iters} completed. Model saved.")

env.close()

# Load the best model
best_model_path = f"{models_dir}/ppo_car_racing_{TIMESTEPS * iters}"
trained_model = PPO.load(best_model_path, device=device)

# Showcase the trained model
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
