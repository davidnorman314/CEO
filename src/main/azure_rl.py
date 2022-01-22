"""Program that downloads reinforcement learning information from Azure blob storage
"""
import argparse
import json
import azure.core.exceptions
from azure_rl.azure_client import AzureClient

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

        for training_id, training_dict in all_trainings.items():
            start_training = training_dict["start_training"]
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
            print(post_train_stats)
            print("")
    elif args.blob_name and args.filename:
        blob = client.get_blob_and_save(args.blob_name, args.filename)
    elif args.blob_name:
        blob = client.get_blob(args.blob_name)

        print(blob)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
