import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


def train(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            "seed": args.seed,
        },
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == "dqn":
        from dqn_agent import DQNAgent

        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(
                checkpoint=torch.load(args.load_checkpoint_path)
            )
            agent_2 = DQNAgent.from_checkpoint(
                checkpoint=torch.load(args.load_checkpoint_path)
            )
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
                epsilon_decay_steps=args.epsilon_decay_episodes,
            )
            agent_2 = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
                epsilon_decay_steps=args.epsilon_decay_episodes,
            )
    agents = [agent, agent_2]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0 and episode > 0:
                eval_env = rlcard.make(
                    args.env,
                    config={
                        "seed": args.seed,
                    },
                )
                eval_env.set_agents(
                    [agent, RandomAgent(num_actions=eval_env.num_actions)]
                )

                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games,
                    )[0],
                )
                print(
                    f"Episode: {episode}/{args.num_episodes} - {episode/args.num_episodes*100:.2f}%"
                )
                env.set_agents(
                    [
                        agent,
                        DQNAgent.from_checkpoint(
                            checkpoint=torch.load(
                                "experiments/no-limit-holdem-dqn/checkpoint_dqn.pt"
                            )
                        ),
                    ]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, "model.pth")
    torch.save(agent, save_path)
    print("Model saved in", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        "--env",
        type=str,
        default="no-limit-holdem",
        choices=[
            "blackjack",
            "leduc-holdem",
            "limit-holdem",
            "doudizhu",
            "mahjong",
            "no-limit-holdem",
            "uno",
            "gin-rummy",
            "bridge",
        ],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=[
            "dqn",
            "nfsp",
        ],
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000000,
    )
    parser.add_argument(
        "--num_eval_games",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="experiments/no-limit-holdem-dqn/",
    )

    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--epsilon_decay_episodes",
        type=int,
        default=200000,
    )

    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)