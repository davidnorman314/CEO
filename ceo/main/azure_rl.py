"""Program that downloads reinforcement learning information from Azure blob storage"""

import argparse
import datetime
import json
import pickle

import azure.core.exceptions
import pandas as pd
from dateutil import parser as dateparser

from ceo.azure_rl.azure_client import AzureClient


def extract_trainings(client: AzureClient, *, earliest_start: datetime.datetime = None):
    """Loads all information from each training returns it."""
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
            all_trainings[training_id]["start"] = dateparser.parse(
                training["start_time"]
            )
        elif training["record_type"] == "end_training":
            all_trainings[training_id]["end_training"] = training
        elif (
            training["record_type"] == "post_train_stats"
            or training["record_type"] == "post_train_test_stats"
        ):
            all_trainings[training_id]["post_train_test_stats"] = training
        elif training["record_type"] == "train_stats":
            if "train_stats" not in all_trainings[training_id]:
                all_trainings[training_id]["train_stats"] = []

            all_trainings[training_id]["train_stats"].append(training)
        else:
            print("Found other", training)

    # Remove the trainings we don't want to process
    all_trainings = {
        k: v for k, v in all_trainings.items() if earliest_start < v["start"]
    }

    # Add information from the logs
    for _training_id, training_dict in all_trainings.items():
        start_training = training_dict["start_training"]

        final_pct_win = None
        post_train_test_stats = None
        if "post_train_test_stats" in training_dict:
            post_train_test_stats = training_dict["post_train_test_stats"]
            final_pct_win = post_train_test_stats["pct_win"]

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

        # Process the log messages for the training.
        max_episode = 0
        max_pct_win = -100.0
        recent_progress_pct_win = None
        log_list = []
        training_dict["log_list"] = log_list
        test_stats_list = []
        training_dict["test_stats_list"] = test_stats_list
        for line in lines:
            line_json = json.loads(line)
            if "record_type" not in line_json or line_json["record_type"] == "log":
                episode = line_json["episode"]
                max_episode = max(episode, max_episode)

                log_list.append(line_json)
            elif line_json["record_type"] == "test_stats":
                test_stats_list.append(line_json)

                pct_win = line_json["pct_win"]
                max_pct_win = max(pct_win, max_pct_win)
                recent_progress_pct_win = pct_win
            elif line_json["record_type"] == "start_training":
                pass
            else:
                print("Unknown", line_json)

        if final_pct_win is None:
            final_pct_win = recent_progress_pct_win
        else:
            max_pct_win = max(final_pct_win, max_pct_win)

        if final_pct_win is None:
            final_pct_win = -1.0
        if max_pct_win is None:
            max_pct_win = -1.0

        training_dict["final_pct_win"] = final_pct_win
        training_dict["max_pct_win"] = max_pct_win
        training_dict["max_episode"] = max_episode

    return all_trainings


def find_feature_def(feature_defs: list, feature_type: str):
    for feature_def in feature_defs:
        type = feature_def[0]
        feature_params = feature_def[1]

        print(type, feature_params)

        if type == feature_type:
            return feature_params

    print("Can't find feature ", feature_type, " from ", feature_defs)

    return None


def get_training_progress(
    client: AzureClient, pickle_file: str, *, earliest_start: datetime.datetime = None
):
    """Loads all information from each trainings and creates a pickle file
    with information about training progress for each one."""

    all_trainings = extract_trainings(client, earliest_start=earliest_start)

    trainings_rows_list = []
    progress_rows_list = []
    features_and_stats = []
    for training_id, training_dict in all_trainings.items():
        start_training = training_dict["start_training"]

        # Add general information about the training
        cols = dict()
        cols["training_id"] = training_id
        cols["learning_type"] = start_training["learning_type"]
        if "action_space_type" in start_training:
            cols["action_space_type"] = start_training["action_space_type"]
        else:
            cols["action_space_type"] = ""
        cols["start"] = pd.to_datetime(start_training["start_time"])
        cols["lambda"] = start_training["params"]["decay"]
        if "max_initial_visit_count" in start_training["params"]:
            cols["max_initial_visit_count"] = start_training["params"][
                "max_initial_visit_count"
            ]
        else:
            cols["max_initial_visit_count"] = None
        cols["discount"] = start_training["params"]["discount_factor"]
        if "alpha_exponent" in start_training["params"]:
            cols["alpha_exponent"] = start_training["params"]["alpha_exponent"]
        else:
            cols["alpha_exponent"] = 0.85

        # Add information about the features used by the agent
        hand_summary = find_feature_def(start_training["feature_defs"], "HandSummary")
        if hand_summary is not None:
            cols["hs_high_card_obs_max"] = hand_summary["high_card_obs_max"]
        else:
            cols["hs_high_card_obs_max"] = None

        # Add end-of-training information
        if "end_training" in training_dict:
            end_training = training_dict["end_training"]
            cols["end"] = pd.to_datetime(end_training["stop_time"])
            cols["finished"] = True
        else:
            cols["end"] = None
            cols["finished"] = False

        if "post_train_test_stats" in training_dict:
            post_train_test_stats = training_dict["post_train_test_stats"]
            cols["final_pct_win"] = post_train_test_stats["pct_win"]
        else:
            cols["final_pct_win"] = None

        max_episode = training_dict["max_episode"]
        cols["max_episode"] = max_episode

        trainings_rows_list.append(cols)

        # Process the progress log messages
        for line_json in training_dict["log_list"]:
            progress_row = dict()

            episode = line_json["episode"]
            max_episode = max(episode, max_episode)

            progress_row["training_id"] = training_id
            progress_row["episode"] = episode
            progress_row["avg_rewards"] = line_json["avg_reward"]
            progress_row["recent_rewards"] = line_json["recent_reward"]
            progress_row["states_visited"] = line_json["states_visited"]
            progress_row["explore_rate"] = line_json["explore_rate"]
            progress_row["pct_win"] = None
            if "skipped_episodes" in line_json:
                progress_row["skipped_episodes"] = line_json["skipped_episodes"]
            else:
                progress_row["skipped_episodes"] = None
            progress_row["recent_rewards"] = line_json["recent_reward"]

            progress_rows_list.append(progress_row)

        # Process the statistics from testing during training
        max_progress_pct_win = -100.0
        for line_json in training_dict["test_stats_list"]:
            progress_row = dict()

            episode = line_json["training_episodes"]
            max_episode = max(episode, max_episode)

            progress_row["training_id"] = training_id
            progress_row["episode"] = episode
            progress_row["avg_rewards"] = None
            progress_row["recent_rewards"] = None
            progress_row["states_visited"] = None
            progress_row["explore_rate"] = None

            pct_win = line_json["pct_win"]
            progress_row["pct_win"] = pct_win
            max_progress_pct_win = max(pct_win, max_progress_pct_win)

            progress_rows_list.append(progress_row)

        final_pct_win = training_dict["final_pct_win"]
        max_pct_win = training_dict["max_pct_win"]

        if final_pct_win is None:
            final_pct_win = -1.0
        if max_pct_win is None:
            max_pct_win = -1.0

        cols["last_pct_win"] = final_pct_win
        cols["max_pct_win"] = max_pct_win

        features_and_stats.append(
            (final_pct_win, max_pct_win, max_episode, start_training)
        )

    # Combine rows for the same episode
    progress_rows_list.sort(key=lambda row: (row["training_id"], row["episode"]))
    fixed_progress_rows_list = []
    for row in progress_rows_list:
        if len(fixed_progress_rows_list) == 0:
            fixed_progress_rows_list.append(row)
            continue

        prev_row = fixed_progress_rows_list[-1]
        # print("prev", prev_row, "row", row)
        if prev_row["episode"] == row["episode"]:
            for key, value in row.items():
                if value is not None:
                    prev_row[key] = value
        else:
            fixed_progress_rows_list.append(row)

    progress_rows_list = fixed_progress_rows_list

    trainings_df = pd.DataFrame(trainings_rows_list)
    progress_df = pd.DataFrame(progress_rows_list)

    data = dict()
    data["trainings"] = trainings_df
    data["progress"] = progress_df

    pickeled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    print("Saving results to", pickle_file)
    with open(pickle_file, "wb") as f:
        f.write(pickeled_data)

    features_and_stats.sort(key=lambda tup: (tup[0], tup[1]))
    for final_pct_win, max_pct_win, episodes, start_training in features_and_stats:
        print(
            "final_pct_win",
            final_pct_win,
            "max_pct_win",
            max_pct_win,
            "episode",
            episodes,
            start_training["learning_type"],
            start_training["training_id"],
            start_training["log_blob_name"],
        )
        for feature_def in start_training["feature_defs"]:
            print("  ", feature_def)


def get_results(client: AzureClient, earliest_start: datetime = None):
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
            all_trainings[training_id]["start"] = dateparser.parse(
                training["start_time"]
            )
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

    for _training_id, training_dict in all_trainings.items():
        if earliest_start is not None and earliest_start > training_dict["start"]:
            continue

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
    parser.add_argument(
        "--earliest-start",
        dest="earliest_start",
        type=str,
        default=None,
        help="Only return trainings that start after the given time.",
    )

    args = parser.parse_args()

    kwargs = dict()
    if args.earliest_start:
        kwargs["earliest_start"] = dateparser.parse(args.earliest_start)
        print("Only returning trainings that started after", kwargs["earliest_start"])

    client = AzureClient()

    if args.get_rl_trainings:
        trainings = client.get_all_trainings()

        for training in trainings:
            print(training)
    elif args.get_results:
        get_results(client, **kwargs)
    elif args.get_training_progress:
        get_training_progress(client, args.get_training_progress, **kwargs)
    elif args.blob_name and args.filename:
        blob = client.get_blob_and_save(args.blob_name, args.filename)
    elif args.blob_name:
        blob = client.get_blob(args.blob_name)

        print(blob)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
