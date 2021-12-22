"""File to setup Azure to support RL training and testing and to run jobs on Azure batch.
   
   The code uses azure.identity.EnvironmentCredential for authentication. This depends on
   the environment variables:
      AZURE_TENANT_ID: In the Azure portal go to the Active Directory section and the Tenant ID
                       will be under Basic information.
      AZURE_CLIENT_ID: The ID of an App Registration under Active Directory.
      AZURE_CLIENT_SECRET: A secret for AZURE_CLIENT_ID, see Certificates and secrets for the 
                           app registration.

   If creating a VM, the AZURE_VM_PASSWORD environment variable must be set.

   If using Azure Batch functionality, then the key for the Azure batch account must be in the
   environment variable AZURE_BATCH_KEY.
"""
import argparse
import json
import os

from azure.identity import AzureCliCredential, EnvironmentCredential

from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    Image,
    SubResource,
    GalleryImage,
    GalleryArtifactVersionSource,
    GalleryImageIdentifier,
    GalleryImageVersion,
    GalleryImageVersionStorageProfile,
    OperatingSystemStateTypes,
    OperatingSystemTypes,
)

import azure.batch._batch_service_client as batch
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels
import azure.batch.models._batch_service_client_enums


class AccountInfo:
    subscription_id: str
    location: str
    resource_group: str
    batch_account: str

    def __init__(self, config: dict):
        self.subscription_id = config["subscription_id"]
        self.location = config["location"]
        self.resource_group = config["resource_group"]
        self.batch_account = config["batch_account"]


def get_batch_vm_images(account_info: AccountInfo, batch_account_key: str):
    """Queries Azure to find out which VM images can be used to create images for
    a batch pool.
    """
    batch_service_url = (
        f"https://{account_info.batch_account}.{account_info.location}.batch.azure.com"
    )

    credentials = batchauth.SharedKeyCredentials(
        account_info.batch_account, batch_account_key
    )

    batch_client = batch.BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Get VM images supported by Azure Batch
    options = batchmodels.AccountListSupportedImagesOptions(
        filter="verificationType eq 'verified'"
    )
    images = list(
        batch_client.account.list_supported_images(
            account_list_supported_images_options=options
        )
    )
    filtered_images = list(
        filter(
            lambda img: img.os_type
            == azure.batch.models._batch_service_client_enums.OSType.linux
            and img.node_agent_sku_id == "batch.node.ubuntu 20.04"
            and img.capabilities is None,
            images,
        )
    )

    print("filtered image count", len(list(filtered_images)))
    for image in filtered_images:
        print(image)

    image = filtered_images[0]
    agent_sku_id = image.node_agent_sku_id
    image_reference = image.image_reference

    print("agent_sku_id", agent_sku_id)
    print("image.sku", image_reference.sku)

    image_reference_json = dict()
    image_reference_json["publisher"] = image_reference.publisher
    image_reference_json["offer"] = image_reference.offer
    image_reference_json["sku"] = image_reference.sku
    image_reference_json["version"] = image_reference.version

    print(json.dumps(image_reference_json, indent=4))

    return agent_sku_id, image_reference


def provision_vm(
    account_info: AccountInfo,
    credential: EnvironmentCredential,
    vm_size: str,
    vm_config: dict,
):
    # Code from https://docs.microsoft.com/en-us/azure/developer/python/azure-sdk-example-virtual-machines?tabs=cmd

    # Create a VM
    print(
        "Provisioning a virtual machine...some operations might take a minute or two."
    )

    # Look up the network information
    network_client = NetworkManagementClient(credential, account_info.subscription_id)
    subnet_result = network_client.subnets.get(
        account_info.resource_group, vm_config["vnet_name"], vm_config["subnet_name"]
    )

    # Provision an IP address and wait for completion
    poller = network_client.public_ip_addresses.begin_create_or_update(
        account_info.resource_group,
        vm_config["ip_address_name"],
        {
            "location": account_info.location,
            "sku": {"name": "Standard"},
            "public_ip_allocation_method": "Static",
            "public_ip_address_version": "IPV4",
        },
    )

    ip_address_result = poller.result()

    print(
        f"Provisioned public IP address {ip_address_result.name} with address {ip_address_result.ip_address}"
    )

    # Provision the network interface client (NIC)
    poller = network_client.network_interfaces.begin_create_or_update(
        account_info.resource_group,
        vm_config["nic_name"],
        {
            "location": account_info.location,
            "ip_configurations": [
                {
                    "name": vm_config["ip_config_name"],
                    "subnet": {"id": subnet_result.id},
                    "public_ip_address": {"id": ip_address_result.id},
                }
            ],
        },
    )

    nic_result = poller.result()

    print(f"Provisioned network interface client {nic_result.name}")

    # Provision the virtual machine
    compute_client = ComputeManagementClient(credential, account_info.subscription_id)

    vm_name = vm_config["name"]
    username = vm_config["admin_username"]
    password = os.getenv("AZURE_VM_PASSWORD")
    if not password or len(password) < 5:
        print("The AZURE_VM_PASSWORD environment variable is not set or is too short.")

    print(
        f"Provisioning virtual machine {vm_name}; this operation might take a few minutes."
    )

    # Provision the VM.
    poller = compute_client.virtual_machines.begin_create_or_update(
        account_info.resource_group,
        vm_name,
        {
            "location": account_info.location,
            "storage_profile": {"image_reference": vm_config["base_image_reference"]},
            "hardware_profile": {"vm_size": vm_size},
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": username,
                "admin_password": password,
            },
            "network_profile": {
                "network_interfaces": [
                    {
                        "id": nic_result.id,
                    }
                ]
            },
        },
    )

    vm_result = poller.result()

    print(f"Provisioned virtual machine {vm_result.name}. Note that the VM is running.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="configfile",
        type=str,
        default="config.json",
        help="The name of the file containing the Azure configuration.",
    )
    parser.add_argument(
        "--query-batch-vm-images",
        dest="get_batch_vm_images",
        action="store_const",
        const=True,
        default=False,
        help="Queries Azure to find VM images that can be used for Batch pools.",
    )
    parser.add_argument(
        "--provision-vm",
        dest="create_vm",
        action="store_const",
        const=True,
        default=False,
        help="Create the VM.",
    )

    args = parser.parse_args()

    print("Loading configuration from", args.configfile)
    with open(args.configfile) as f:
        config = json.load(f)

    account_info = AccountInfo(config)
    vm_size = config["vm_size"]

    # Create a credential object from the environment.
    credential = EnvironmentCredential()

    batch_account_key = None
    if args.get_batch_vm_images:
        batch_key_env_var = "AZURE_BATCH_KEY"
        batch_account_key = os.getenv(batch_key_env_var)

        if not batch_account_key:
            raise Exception("Environment variable", batch_key_env_var, "is not set.")

    if args.get_batch_vm_images:
        assert batch_account_key is not None
        get_batch_vm_images(account_info, batch_account_key)
    if args.create_vm:
        provision_vm(account_info, credential, vm_size, config["vm_config"])


if __name__ == "__main__":
    # execute only if run as a script
    main()
