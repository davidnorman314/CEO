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

    def start_training(self, learning_type: str, params: dict):
        self._training_id = "tid_" + str(uuid.uuid4())

        desc = dict()
        desc["learning_type"] = learning_type
        desc["start_time"] = datetime.datetime.now().isoformat()
        desc["log_blob_name"] = self.log_blob_name
        desc["pickle_blob_name"] = self.pickle_blob_name
        desc["params"] = params
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

    def get_all_trainings(self):
        rl_trainings_blob_client = self.container_client.get_blob_client(
            self.rl_trainings_blob_name
        )

        downloader = rl_trainings_blob_client.download_blob()
        data = downloader.readall()
        data_str = data.decode("utf-8")

        print(type(data))
        print(type(data_str))
        return data_str.split("\n")


if __name__ == "__main__":
    # Test method
    params = dict()
    params["p1"] = "v1"
    params["p2"] = "v2"

    azure_blob = AzureClient()
    azure_blob.start_training("main_test", params)
