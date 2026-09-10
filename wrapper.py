import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv


from pyboy import PyBoy, WindowEvent

filename = "ROMs/mario.gb"

pyboy = PyBoy(filename, window_type="SDL2", game_wrapper=True)
eval_pyboy = PyBoy(filename, window_type="headless", game_wrapper=True, disable_renderer=True)
pyboy.game_wrapper().start_game()
pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

pyboy.set_emulation_speed(0)
eval_pyboy.set_emulation_speed(0)

assert pyboy.cartridge_title() == "SUPER MARIOLAN"
assert eval_pyboy.cartridge_title() == "SUPER MARIOLAN"

env = pyboy.openai_gym(observation_type="raw", action_type="all")

eval_env = DummyVecEnv([lambda: Monitor(eval_pyboy.openai_gym(observation_type="raw", action_type="all"))])
eval_env = VecTransposeImage(eval_env)

best_model_path = "logs/best_model.zip"
if os.path.exists(best_model_path):
    print("Loading best model...")
    model = PPO.load(best_model_path, env=env)
else:
    # Custom MLP policy with more layers
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )

    model = PPO('MlpPolicy', env, verbose=1, device="mps", policy_kwargs=policy_kwargs,
                learning_rate=2.5e-4, n_steps=8192, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95)
'''
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                         name_prefix='rl_model')

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/',
                             eval_freq=1600, n_eval_episodes=1, deterministic=True)

model.learn(total_timesteps=int(4e5), callback=[eval_callback])

del model
'''

# Load the trained agent
model = PPO.load("logs/best_model", device="mps")

# Enjoy trained agent
obs = env.reset()
done = False
rewards = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

print(f"Final reward: {rewards}")
