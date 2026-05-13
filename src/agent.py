import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        min_buffer_to_train=1000,
        target_update_freq=500,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=50_000,
        eps_schedule="linear",   # "linear", "exponential", "step"
        eps_decay_rate=0.9995,   # pour exponential
        eps_step_at=20_000,      # pour step
        double_dqn=False,
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_to_train = min_buffer_to_train
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_schedule = eps_schedule
        self.eps_decay_rate = eps_decay_rate
        self.eps_step_at = eps_step_at
        self.double_dqn = double_dqn
        self.device = device

        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # huber, plus stable
        self.buffer = ReplayBuffer(buffer_size)

        self.step_count = 0
        self.eps = eps_start

    def epsilon(self):
        if self.eps_schedule == "linear":
            frac = min(1.0, self.step_count / self.eps_decay_steps)
            return self.eps_start + frac * (self.eps_end - self.eps_start)
        elif self.eps_schedule == "exponential":
            return max(self.eps_end, self.eps_start * (self.eps_decay_rate ** self.step_count))
        elif self.eps_schedule == "step":
            return self.eps_start if self.step_count < self.eps_step_at else self.eps_end
        else:
            raise ValueError(self.eps_schedule)

    def act(self, state, greedy=False):
        if greedy:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return int(self.q_net(s).argmax(dim=1).item())
        self.eps = self.epsilon()
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def train_step(self):
        if len(self.buffer) < self.min_buffer_to_train:
            return None

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        s_next = torch.from_numpy(s_next).to(self.device)
        done = torch.from_numpy(done).to(self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # online choisit l'action, target évalue
                next_actions = self.q_net(s_next).argmax(dim=1, keepdim=True)
                next_q = self.target_net(s_next).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(s_next).max(dim=1)[0]
            target = r + self.gamma * next_q * (1.0 - done)

        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        # on évite que ça explose
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def increment_step(self):
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.step_count = ckpt.get("step_count", 0)
