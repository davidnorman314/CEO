"""File for evaluating parsing the PPO eval_log."""

import argparse
import pathlib
import json
import numpy as np
import pandas as pd
import tabulate


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
    print(" ")
    print("   Pct win (overall)", win_count / (win_count + loss_count + invalid_action_count))
    print("Pct win (only valid)", win_count / (win_count + loss_count))
    print(" ")

    ep_sum_lengths = 0.0
    count = 0.0
    for ep_length in last_ep_lengths:
        ep_sum_lengths += ep_length
        count += 1.0

    print("Avg episode length", ep_sum_lengths / count)


def load_all_eval(eval_dirs: list[str]):
    for path_str in eval_dirs:
        path = pathlib.Path(path_str)

        name = []
        num_players = []
        seat_number = []
        avg_rewards = []
        for subdir in path.iterdir():
            evaluations = np.load(subdir / "evaluations.npz")
            param_file = subdir / "params.json"

            if param_file.exists():
                with open(param_file, "r") as data_file:
                    params = json.load(data_file)

                this_num_players = params["env_args"]["num_players"]
                this_seat_number = params["env_args"]["seat_number"]
            else:
                this_num_players = 6
                this_seat_number = 0

            # Take the average of the last 10 evaluations, assuming that the agent training has
            # reached steady state
            eval_reward = []
            for eval in evaluations["results"][-11:-1]:
                eval_reward.append(np.average(eval))
            avg_reward = np.average(eval_reward)

            name.append(subdir.stem)
            num_players.append(this_num_players)
            seat_number.append(this_seat_number)
            avg_rewards.append(avg_reward)

    df = pd.DataFrame(
        {
            "Num Players": num_players,
            "Seat Number": seat_number,
            "Avg Reward": avg_rewards,
        },
        index=name,
    )
    df.sort_values(
        by=["Num Players", "Seat Number", "Avg Reward"], ascending=[True, True, False], inplace=True
    )

    print(tabulate.tabulate(df, headers="keys", tablefmt="github"))


# Main function
def main():
    parser = argparse.ArgumentParser(description="Do learning")

    parser.add_argument(
        "--eval-dir",
        dest="eval_dir",
        default=None,
        help="The directory containing the evaluation information.",
    )
    parser.add_argument(
        "--all",
        dest="all",
        type=str,
        nargs="*",
        default=[],
        help="Load all logs in the given directories.",
    )

    args = parser.parse_args()

    if args.eval_dir:
        load_eval(args.eval_dir)
    elif args.all:
        load_all_eval(args.all)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
