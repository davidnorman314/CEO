"""Program that downloads reinforcement learning information from Azure blob storage
"""
import argparse
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
    elif args.blob_name and args.filename:
        blob = client.get_blob_and_save(args.blob_name, args.filename)
    elif args.blob_name:
        blob = client.get_blob(args.blob_name)

        print(blob)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()