"""File for evaluating PPO training"""

import argparse
import numpy as np


def load_eval(eval_dir: str):
    print("Loading", eval_dir)

    evaluations = np.load(eval_dir + "/evaluations.npz")
    if False:
        print("evaluations", evaluations)
        print("evaluations.files", evaluations.files)
        print("evaluations['timesteps']", evaluations["timesteps"])
        print("evaluations['results']", evaluations["results"])
        print("evaluations['ep_lengths']", evaluations["ep_lengths"])

    last_results = evaluations["results"][-1]
    last_ep_lengths = evaluations["ep_lengths"][-1]

    win_count = 0
    loss_count = 0
    invalid_action_count = 0
    for reward in last_results:
        if reward == 1.0:
            win_count += 1
        elif reward == -1.0:
            loss_count += 1
        elif -10.0 <= reward <= 2.0:
            invalid_action_count += 1
        else:
            raise Exception("Unknown reward:", reward)

    print("   Wins", win_count)
    print(" Losses", loss_count)
    print("Invalid", invalid_action_count)

    ep_sum_lengths = 0.0
    count = 0.0
    for ep_length in last_ep_lengths:
        ep_sum_lengths += ep_length
        count += 1.0

    print("Avg episode length", ep_sum_lengths / count)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Do learning")

    parser.add_argument(
        "eval_dir",
        help="The directory containing the evaluation information.",
    )

    args = parser.parse_args()

    load_eval(args.eval_dir)


if __name__ == "__main__":
    main()
