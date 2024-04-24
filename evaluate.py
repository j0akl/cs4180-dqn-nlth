""" An example of evluating the trained models in RLCard
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
)
from modified_tournament import tournament


def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch

        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent

        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == "random":  # Random model
        from rlcard.agents import RandomAgent

        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models

        agent = models.load(model_path).agents[position]

    return agent


def evaluate(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={"seed": args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards, tracked_profit = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)
    return {
        args.models[position]: reward for position, reward in enumerate(rewards)
    }, tracked_profit


def evaluate_over_number_seeds(args, num_runs):
    avg_reward = {}
    tracked_profits = []
    for seed in range(num_runs):
        args.seed = seed
        rewards, profit = evaluate(args)
        tracked_profits.append(profit)
        if avg_reward == {}:
            avg_reward = {
                args.models[position]: [] for position, reward in enumerate(rewards)
            }

        for key, val in rewards.items():
            avg_reward[key].append(val)
    """
    for key, val in avg_reward:
        print(f"{key}: {sum(val)/len(val)}")
    """

    plot_results(tracked_profits)


def plot_results(tracked_profits):
    df = pd.DataFrame(tracked_profits)
    df = df.cumsum(axis=1)
    means = df.mean()
    std_devs = df.std()
    std_devs = std_devs / np.sqrt(df.shape[0]) * 1.96
    # Set up the plot
    plt.figure(figsize=(10, 5))

    x = np.arange(len(means))

    plt.plot(x, means, label="Average Profit")

    plt.fill_between(x, means - std_devs, means + std_devs, color="blue", alpha=0.2)

    # Adding titles and labels
    plt.title("Profit against Random Agent")
    plt.xlabel("Game Number")
    plt.ylabel("Average Profit")

    # Adding legend
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

    # Show the plot
    plt.savefig("results/ai_vs_random.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
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
        ],
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "experiments/no-limit-holdem-dqn/model.pth",
            # "experiments/no-limit-holdem-dqn/model.pth",
            "random",
        ],
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.num_seeds > 0:
        evaluate_over_number_seeds(args, args.num_seeds)
    else:
        evaluate(args)
