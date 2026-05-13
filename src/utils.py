import os
import random
import numpy as np
import torch
import imageio
import gymnasium as gym


def _resolve_env_id():
    # v3 sur gymnasium >= 1.0, fallback v2 sinon
    for eid in ("LunarLander-v3", "LunarLander-v2"):
        try:
            gym.spec(eid)
            return eid
        except Exception:
            continue
    return "LunarLander-v3"


LUNAR_ENV_ID = _resolve_env_id()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id=None, render_mode=None, seed=None):
    env_id = env_id or LUNAR_ENV_ID
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def record_episode(agent, out_path, env_id=None, seed=0, max_steps=1000, fps=30):
    env = gym.make(env_id or LUNAR_ENV_ID, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        action = agent.act(state, greedy=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        if out_path.endswith(".gif"):
            imageio.mimsave(out_path, frames, fps=fps, loop=0)
        else:
            # mp4 -> nécessite imageio-ffmpeg, sinon on retombe en gif
            try:
                imageio.mimsave(out_path, frames, fps=fps, codec="libx264")
            except Exception:
                fallback = os.path.splitext(out_path)[0] + ".gif"
                imageio.mimsave(fallback, frames, fps=fps, loop=0)
    except Exception as e:
        print(f"[record_episode] skip {out_path}: {e}")
    return total_reward


def evaluate(agent, env_id=None, n_episodes=10, seed=123, max_steps=1000):
    env = gym.make(env_id or LUNAR_ENV_ID)
    rewards = []
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed + i)
        total = 0.0
        for _ in range(max_steps):
            a = agent.act(s, greedy=True)
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return np.mean(rewards), np.std(rewards), rewards
