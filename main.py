from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from pyboy import PyBoy


def make_mario_env(window_type="SDL2"):
    filename = "ROMs/mario.gb"
    pyboy = PyBoy(filename, window_type=window_type, game_wrapper=True)
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title() == "SUPER MARIOLAN"
    return pyboy.openai_gym(observation_type="raw", action_type="all")

# Number of parallel environments
n_envs = 1

# Create vectorized environment
env = make_vec_env(make_mario_env, n_envs=n_envs)

# Correct format for net_arch following SB3 v1.8.0 update
policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))

# Function to create evaluation environment wrapped with Monitor
def make_eval_env():
    eval_env = make_mario_env("headless")
    eval_env = Monitor(eval_env)  # Wrap with Monitor for correct statistics
    eval_env = DummyVecEnv([lambda: eval_env])  # Wrap with DummyVecEnv for compatibility
    eval_env = VecTransposeImage(eval_env)  # Apply VecTransposeImage if needed
    return eval_env

# Callbacks for checkpointing and evaluations
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='rl_model')
eval_callback = EvalCallback(make_eval_env(), best_model_save_path='./logs/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)

model = PPO('MlpPolicy', env, verbose=1, device="mps", policy_kwargs=policy_kwargs, learning_rate=2.5e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95)

model.learn(total_timesteps=int(2e5), callback=[checkpoint_callback, eval_callback])

model.save("ppo_mario")
del model  # remove to demonstrate saving and loading

# Load the trained agent
model = PPO.load("ppo_mario")

# Enjoy trained agent
obs = env.reset()
done = [False for _ in range(n_envs)]
rewards = [0 for _ in range(n_envs)]
while not all(done):
    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(actions)

print(f"Final rewards: {rewards}")
