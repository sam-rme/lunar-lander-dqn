"""
Script d'entraînement simple. Lance avec:
    python train.py
    python train.py --double-dqn
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from tqdm import trange

from src.agent import DQNAgent
from src.utils import set_seed, make_env, record_episode, evaluate, LUNAR_ENV_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--eps-schedule", default="linear", choices=["linear", "exponential", "step"])
    p.add_argument("--target-update-freq", type=int, default=500)
    p.add_argument("--save-name", default=None)
    p.add_argument("--record-every", type=int, default=200,
                   help="enregistre une vidéo tous les N épisodes (0 pour désactiver)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    env = make_env(seed=args.seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        eps_schedule=args.eps_schedule,
        target_update_freq=args.target_update_freq,
        double_dqn=args.double_dqn,
        device=device,
    )

    name = args.save_name or ("double_dqn" if args.double_dqn else "dqn")
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    rewards = []
    losses = []
    window = deque(maxlen=100)
    best_avg = -np.inf
    t0 = time.time()

    pbar = trange(args.episodes)
    for ep in pbar:
        s, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        ep_losses = []

        for _ in range(args.max_steps):
            a = agent.act(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.push(s, a, r, s_next, float(terminated))  # truncated != terminal vrai
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            agent.increment_step()
            s = s_next
            total_reward += r
            if done:
                break

        rewards.append(total_reward)
        window.append(total_reward)
        avg = np.mean(window)
        if ep_losses:
            losses.append(np.mean(ep_losses))

        pbar.set_description(
            f"ep {ep} | r {total_reward:7.1f} | avg100 {avg:7.1f} | eps {agent.eps:.3f}"
        )

        # checkpoint si on bat le record et qu'on a un buffer suffisant
        if len(window) >= 50 and avg > best_avg:
            best_avg = avg
            agent.save(f"outputs/checkpoints/{name}_best.pt")

        if args.record_every and ep > 0 and ep % args.record_every == 0:
            record_episode(agent, f"outputs/videos/{name}_ep{ep}.gif", seed=args.seed + 9999)

        # early stop si problème "résolu"
        if avg >= 200 and len(window) == 100:
            print(f"\nresolved at episode {ep} (avg100 = {avg:.1f})")
            break

    agent.save(f"outputs/checkpoints/{name}_final.pt")
    np.save(f"outputs/checkpoints/{name}_rewards.npy", np.array(rewards))

    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min")

    # eval finale
    mean_r, std_r, _ = evaluate(agent, n_episodes=20, seed=999)
    print(f"eval: {mean_r:.1f} ± {std_r:.1f}")

    record_episode(agent, f"outputs/videos/{name}_final.gif", seed=42)
    record_episode(agent, f"outputs/videos/{name}_final.mp4", seed=42)


if __name__ == "__main__":
    main()
