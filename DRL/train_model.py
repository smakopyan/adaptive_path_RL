import argparse
from dataclasses import dataclass

import numpy as np
import rclpy
import torch

from .environment import Environment, EnvConfig, RosConfig
from .TD3 import TD3
from navigation_src.navigation_node import *


@dataclass
class ReplayBuffer:
    state_dim: int
    action_dim: int
    max_size: int = 300_000

    def __post_init__(self):
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx]


def run(args):
    rclpy.init(args=None)

    env_cfg = EnvConfig(max_steps=args.max_steps, dt=args.dt)
    ros_cfg = RosConfig(
        map_topic=args.map_topic,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        reset_service=args.reset_service,
    )
    env = Environment(config=env_cfg, ros=ros_cfg)

    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_size=256,
            device=("cuda" if args.cuda and torch.cuda.is_available() else "cpu"),
            batch_size=args.batch_size,
        )

        replay = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size)

        total_steps = 0
        for ep in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0.0

            for _ in range(args.max_steps):
                total_steps += 1
                if total_steps < args.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.get_action(state, training=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.add(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward

                if replay.size >= args.batch_size:
                    agent.train(replay)

                if done:
                    break

            print(
                f"episode={ep:04d} reward={ep_reward:.3f} "
                f"dist={info.get('dist', -1):.2f} min_scan={info.get('min_scan', -1):.2f} steps={total_steps}"
            )

        if args.model_out:
            torch.save(agent.actor.state_dict(), args.model_out)
            print(f"saved actor to {args.model_out}")
    finally:
        env.close()
        rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description="Train TD3 residual policy over ROS topics (/map,/odom,/scan,/cmd_vel)")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--buffer-size", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--model-out", default="DRL/models/td3_actor.pt")

    p.add_argument("--map-topic", default="/map")
    p.add_argument("--odom-topic", default="/odom")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--cmd-vel-topic", default="/cmd_vel")
    p.add_argument("--reset-service", default="/reset_world")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
