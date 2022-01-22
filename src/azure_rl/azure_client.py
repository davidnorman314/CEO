import os
import uuid
import datetime
import json
from azure.storage.blob import (
    BlobProperties,
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    BlobType,
    __version__,
)


class AzureClient:

    connection_env_var = "AZURE_STORAGE_CONNECTION_STRING"
    rl_trainings_blob_name = "rl_trainings"
    container_name = "ceorl"

    connect_str: str

    log_blob_name: str
    pickle_blob_name: str

    blob_service_client: BlobServiceClient
    container_client: ContainerClient
    log_client: BlobClient

    _training_id: str

    def __init__(self):
        self.connect_str = os.getenv(self.connection_env_var)
        if not self.connect_str:
            raise Exception("Environment variable", self.connection_env_var, "is not set.")

        self.log_blob_name = "log_" + str(uuid.uuid4())
        self.pickle_blob_name = "pkl_" + str(uuid.uuid4())

        # Connect to Azure
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def start_training(
        self, learning_type: str, player_count: int, params: dict, feature_defs: list
    ):
        self._training_id = "tid_" + str(uuid.uuid4())

        desc = dict()
        desc["record_type"] = "start_training"
        desc["learning_type"] = learning_type
        desc["start_time"] = datetime.datetime.now().isoformat()
        desc["log_blob_name"] = self.log_blob_name
        desc["pickle_blob_name"] = self.pickle_blob_name
        desc["player_count"] = player_count
        desc["params"] = params
        desc["feature_defs"] = feature_defs
        desc["training_id"] = self._training_id

        # Use ndjson format
        json_str = json.dumps(desc, separators=(",", ":"), indent=None)
        json_str = json_str + "\n"

        blob_properties = BlobProperties(
            name=self.rl_trainings_blob_name, blob_type=BlobType.AppendBlob
        )
        rl_trainings_blob_client = self.container_client.get_blob_client(blob_properties)

        if not rl_trainings_blob_client.exists():
            rl_trainings_blob_client.create_append_blob()
        rl_trainings_blob_client.append_block(json_str, len(json_str))

        # Connect to the blob that will contain log messages
        self.log_client = self.container_client.get_blob_client(self.log_blob_name)

        self.log_client.create_append_blob()
        self.log_client.append_block(json_str, len(json_str))

        print("Saving information to Azure blob storage")

    def end_training(self):
        desc = dict()
        desc["record_type"] = "end_training"
        desc["stop_time"] = datetime.datetime.now().isoformat()
        desc["training_id"] = self._training_id

        # Use ndjson format
        json_str = json.dumps(desc, separators=(",", ":"), indent=None)
        json_str = json_str + "\n"

        rl_trainings_blob_client = self.container_client.get_blob_client(
            self.rl_trainings_blob_name
        )

        rl_trainings_blob_client.append_block(json_str, len(json_str))

    def log(self, stats: dict):
        """Log information about the search"""

        # Convert any datetimes or timedeltas to strings
        new_stats = dict()
        for key, value in stats.items():
            if isinstance(value, datetime.datetime):
                new_stats[key] = value.isoformat()
            elif isinstance(value, datetime.timedelta):
                new_stats[key] = str(value)
            else:
                new_stats[key] = value

        # Use ndjson format
        json_str = json.dumps(new_stats, separators=(",", ":"), indent=None)
        json_str = json_str + "\n"

        self.log_client.append_block(json_str, len(json_str))

    def save_post_train_stats(
        self, *, episodes=None, total_wins=None, total_losses=None, pct_win=None
    ):
        """Save statistics from running the agent after it is trained."""

        stats = dict()
        stats["record_type"] = "post_train_stats"
        stats["episodes"] = episodes
        stats["total_wins"] = total_wins
        stats["total_losses"] = total_losses
        stats["pct_win"] = pct_win
        stats["training_id"] = self._training_id
        stats["test_time"] = datetime.datetime.now().isoformat()

        # Use ndjson format
        json_str = json.dumps(stats, separators=(",", ":"), indent=None)
        json_str = json_str + "\n"

        rl_trainings_blob_client = self.container_client.get_blob_client(
            self.rl_trainings_blob_name
        )

        rl_trainings_blob_client.append_block(json_str, len(json_str))

    def upload_pickle(self, *, filename: str = None, data: bytes = None):
        # Connect to the blob for the pickled information
        blob_client = self.container_client.get_blob_client(self.pickle_blob_name)

        if filename is not None:
            assert data is None
            with open(filename, "rb") as f:
                data = f.read()

        print("Uploading pickled information to Azure")
        blob_client.upload_blob(data, overwrite=True, length=len(data))
        print("Done uploading pickled information to Azure")

    def get_all_trainings(self):
        rl_trainings_blob_client = self.container_client.get_blob_client(
            self.rl_trainings_blob_name
        )

        downloader = rl_trainings_blob_client.download_blob()
        data = downloader.readall()
        data_str = data.decode("utf-8")

        return data_str.split("\n")

    def get_blob(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)

        downloader = blob_client.download_blob()
        data = downloader.readall()
        data_str = data.decode("utf-8")

        return data_str

    def get_blob_raw(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)

        downloader = blob_client.download_blob()
        data = downloader.readall()

        return data

    def get_blob_and_save(self, blob_name, filename):
        blob_client = self.container_client.get_blob_client(blob_name)

        downloader = blob_client.download_blob()
        data = downloader.readall()

        with open(filename, "wb") as out:
            out.write(data)


if __name__ == "__main__":
    # Test method
    params = dict()
    params["p1"] = "v1"
    params["p2"] = "v2"

    azure_blob = AzureClient()
    azure_blob.start_training("main_test", params)
