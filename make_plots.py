"""Génère les plots à partir des rewards sauvegardés."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
os.makedirs('outputs/plots', exist_ok=True)


def moving_avg(x, w=100):
    x = np.asarray(x)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def plot_one(name, w=100):
    path = f'outputs/checkpoints/{name}_rewards.npy'
    if not os.path.exists(path):
        print(f'skip {name}: pas de rewards trouvés')
        return
    rewards = np.load(path)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.3, label='episode reward')
    ma = moving_avg(rewards, w)
    ax.plot(np.arange(len(ma)) + w, ma, label=f'moving avg ({w})', linewidth=2)
    ax.axhline(200, color='r', linestyle='--', alpha=0.5, label='solved')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.set_title(f'{name.upper()} — training reward')
    ax.legend()
    plt.tight_layout()
    out = f'outputs/plots/{name}_training.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'-> {out}  (final avg{w}={np.mean(rewards[-w:]):.1f}, n_ep={len(rewards)})')


def plot_compare():
    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False
    for tag, color in [('dqn', '#1f77b4'), ('double_dqn', '#d62728')]:
        path = f'outputs/checkpoints/{tag}_rewards.npy'
        if not os.path.exists(path):
            continue
        r = np.load(path)
        ma = moving_avg(r, 100)
        ax.plot(np.arange(len(ma)) + 100, ma, label=tag.upper(), color=color, linewidth=2)
        any_plotted = True
    if not any_plotted:
        print('rien à comparer'); return
    ax.axhline(200, color='r', linestyle='--', alpha=0.5, label='solved')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward (MA100)')
    ax.set_title('DQN vs Double DQN')
    ax.legend()
    plt.tight_layout()
    out = 'outputs/plots/dqn_vs_double.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'-> {out}')


if __name__ == '__main__':
    plot_one('dqn')
    plot_one('double_dqn')
    plot_compare()
