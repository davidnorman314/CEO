"""Program that downloads reinforcement learning information from Azure blob storage
"""
import argparse
import json
import pickle
import azure.core.exceptions

import pandas as pd
import numpy as np

from azure_rl.azure_client import AzureClient


def get_training_progress(client: AzureClient, pickle_file: str):
    """Loads all information from each traning and creates a pickle file
    with information about training progress for each one."""
    trainings = client.get_all_trainings()

    all_trainings = dict()
    for training_str in trainings:
        if len(training_str) == 0:
            continue

        training = json.loads(training_str)
        training_id = training["training_id"]

        if training["record_type"] == "start_training":
            all_trainings[training_id] = dict()
            all_trainings[training_id]["start_training"] = training
        elif training["record_type"] == "end_training":
            all_trainings[training_id]["end_training"] = training
        elif training["record_type"] == "post_train_stats":
            all_trainings[training_id]["post_train_test_stats"] = training
        elif training["record_type"] == "post_train_test_stats":
            all_trainings[training_id]["post_train_test_stats"] = training
        elif training["record_type"] == "train_stats":
            if "train_stats" not in all_trainings[training_id]:
                all_trainings[training_id]["train_stats"] = []

            all_trainings[training_id]["train_stats"].append(training)
        else:
            print("Found other", training)

    trainings_rows_list = []
    progress_rows_list = []
    features_and_stats = []
    for training_id, training_dict in all_trainings.items():
        start_training = training_dict["start_training"]

        cols = dict()
        cols["training_id"] = training_id
        cols["learning_type"] = start_training["learning_type"]
        cols["start"] = pd.to_datetime(start_training["start_time"])

        if "end_training" in training_dict:
            end_training = training_dict["end_training"]
            cols["end"] = pd.to_datetime(end_training["stop_time"])
            cols["finished"] = True
        else:
            cols["end"] = None
            cols["finished"] = False

        train_stats = None
        if "train_stats" in training_dict:
            train_stats = training_dict["train_stats"]

        final_pct_win = None
        if "post_train_test_stats" in training_dict:
            post_train_test_stats = training_dict["post_train_test_stats"]
            final_pct_win = post_train_test_stats["pct_win"]
            cols["final_pct_win"] = final_pct_win
        else:
            cols["final_pct_win"] = None

        trainings_rows_list.append(cols)

        # Get the log messages
        log_blob_name = start_training["log_blob_name"]

        try:
            blob = client.get_blob(log_blob_name)
        except azure.core.exceptions.ResourceNotFoundError:
            print(f"Blob {log_blob_name} does not exist")
            continue

        lines = blob.split("\n")
        if len(lines[-1]) == 0:
            lines.pop()

        # Process the train stats from the log
        all_test_stats = dict()

        for line in lines:
            line_json = json.loads(line)
            if "record_type" not in line_json:
                continue

            if line_json["record_type"] == "test_stats":
                all_test_stats[line_json["training_episodes"]] = line_json["pct_win"]

        # Process the train stats for the training. TODO: The stats have been moved to the
        # log file, so this can be removed.
        if "train_stats" in training_dict:
            for train_stats in training_dict["train_stats"]:
                all_test_stats[train_stats["training_episodes"]] = train_stats["pct_win"]

        # Process the log messages for the training.
        max_progress_pct_win = -1.0
        max_episode = -1
        for line in lines:
            line_json = json.loads(line)
            if "record_type" not in line_json or line_json["record_type"] == "log":
                progress_row = dict()

                episode = line_json["episode"]
                max_episode = max(episode, max_episode)

                progress_row["training_id"] = training_id
                progress_row["episode"] = episode
                progress_row["avg_rewards"] = line_json["avg_reward"]
                progress_row["recent_rewards"] = line_json["recent_reward"]
                progress_row["states_visited"] = line_json["states_visited"]
                progress_row["explore_rate"] = line_json["explore_rate"]

                if episode in all_test_stats:
                    pct_win = all_test_stats[episode]
                    progress_row["pct_win"] = pct_win
                    max_progress_pct_win = max(pct_win, max_progress_pct_win)
                else:
                    progress_row["pct_win"] = None

                progress_rows_list.append(progress_row)

        if final_pct_win is None:
            final_pct_win = max_progress_pct_win

        features_and_stats.append((final_pct_win, max_episode, start_training))

    trainings_df = pd.DataFrame(trainings_rows_list)
    progress_df = pd.DataFrame(progress_rows_list)

    data = dict()
    data["trainings"] = trainings_df
    data["progress"] = progress_df

    pickeled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    print("Saving results to", pickle_file)
    with open(pickle_file, "wb") as f:
        f.write(pickeled_data)

    features_and_stats.sort()
    for pct_win, episodes, start_training in features_and_stats:
        print(
            "pct_win",
            pct_win,
            "episode",
            episode,
            start_training["learning_type"],
            start_training["training_id"],
        )
        for feature_def in start_training["feature_defs"]:
            print("  ", feature_def)


def get_results(client: AzureClient):
    trainings = client.get_all_trainings()

    all_trainings = dict()
    for training_str in trainings:
        if len(training_str) == 0:
            continue

        training = json.loads(training_str)

        if training["record_type"] == "start_training":
            training_id = training["training_id"]
            all_trainings[training_id] = dict()
            all_trainings[training_id]["start_training"] = training
        elif training["record_type"] == "post_train_stats":
            training_id = training["training_id"]
            all_trainings[training_id]["post_train_stats"] = training
        elif training["record_type"] == "train_stats":
            training_id = training["training_id"]
            all_trainings[training_id]["train_stats"] = training
        elif training["record_type"] == "end_training":
            training_id = training["training_id"]
            all_trainings[training_id]["end_training"] = training
        else:
            print("Unknown", training)

    for training_id, training_dict in all_trainings.items():
        start_training = training_dict["start_training"]

        train_stats = None
        if "train_stats" in training_dict:
            train_stats = training_dict["train_stats"]

        post_train_stats = None
        if "post_train_stats" in training_dict:
            post_train_stats = training_dict["post_train_stats"]

        blob_name = start_training["log_blob_name"]

        try:
            blob = client.get_blob(blob_name)
        except azure.core.exceptions.ResourceNotFoundError:
            print(f"Blob {blob_name} does not exist")
            continue

        lines = blob.split("\n")

        i = -1
        if len(lines[i]) == 0:
            i = -2

        print(start_training)
        print(lines[i])
        print(train_stats)
        print(post_train_stats)
        print("")


# Main function
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--get-rl-trainings",
        dest="get_rl_trainings",
        action="store_const",
        const=True,
        default=False,
        help="Download the list of all RL trainings.",
    )
    parser.add_argument(
        "--get-results",
        dest="get_results",
        action="store_const",
        const=True,
        default=False,
        help="Returns the final results of all trainings.",
    )
    parser.add_argument(
        "--get-training-progress",
        dest="get_training_progress",
        type=str,
        default=None,
        help="Downloads information about the progress over time of each training "
        + "and saves to a pickle file. The argument is the name of the pickle file.",
        metavar="PICKLE_FILE",
    )
    parser.add_argument(
        "--get-blob",
        dest="blob_name",
        type=str,
        default=None,
        help="The name of the blob to download.",
    )
    parser.add_argument(
        "--save-file",
        dest="filename",
        type=str,
        default=None,
        help="The name of the file where the downloaded data should be saved.",
    )

    args = parser.parse_args()

    client = AzureClient()

    if args.get_rl_trainings:
        trainings = client.get_all_trainings()

        for training in trainings:
            print(training)
    elif args.get_results:
        get_results(client)
    elif args.get_training_progress:
        get_training_progress(client, args.get_training_progress)
    elif args.blob_name and args.filename:
        blob = client.get_blob_and_save(args.blob_name, args.filename)
    elif args.blob_name:
        blob = client.get_blob(args.blob_name)

        print(blob)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
