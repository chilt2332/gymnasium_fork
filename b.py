from stable_baselines3 import PPO
from gymnasium.envs.registration import make



best_model_path = f"models4/PPO/CnnPolicy/60000.zip"
trained_model = PPO.load(best_model_path, device="cpu")


# Showcase the trained model
env = make('CarRacing-v3', render_mode='human')
obs, _ = env.reset()

done = False
total_reward = 0

# Run the trained model and render the environment
while not done:
    action, _ = trained_model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()  # Render the environment (this should show the car on the screen)

env.close()

print(f"Showcase completed with total reward: {total_reward}")