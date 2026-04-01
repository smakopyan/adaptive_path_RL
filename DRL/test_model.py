import argparse

import numpy as np
import rclpy
import torch

from .environment import Environment, EnvConfig, RosConfig
from .TD3 import TD3
from navigation_src.navigation_node import *


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

        device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

        agent = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_size=256,
            device=device,
            batch_size=args.batch_size,
        )

        actor_state = torch.load(args.model_path, map_location=device)
        agent.actor.load_state_dict(actor_state)
        agent.actor.eval()

        for ep in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0.0
            info = {}

            for step in range(1, args.max_steps + 1):
                with torch.no_grad():
                    action = agent.get_action(state, training=False)

                next_state, reward, terminated, truncated, info = env.step(action)

                state = next_state
                ep_reward += reward

                if args.print_every > 0 and step % args.print_every == 0:
                    print(
                        f"[episode {ep:04d} step {step:04d}] "
                        f"reward={ep_reward:.3f} "
                        f"dist={info.get('dist', -1):.2f} "
                        f"min_scan={info.get('min_scan', -1):.2f}"
                    )

                if terminated or truncated:
                    break

            print(
                f"episode={ep:04d} total_reward={ep_reward:.3f} "
                f"dist={info.get('dist', -1):.2f} "
                f"min_scan={info.get('min_scan', -1):.2f} "
                f"steps={step} "
                f"terminated={terminated} truncated={truncated}"
            )

    finally:
        env.close()
        rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description="Test TD3 policy over ROS topics (/map,/odom,/scan,/cmd_vel)")
    p.add_argument("--model-path", default="DRL/models/td3_actor_AL.pt", help="Path to saved actor weights")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--print-every", type=int, default=100)

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