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
from azure.common.credentials import ServicePrincipalCredentials

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


def get_batch_vm_images(
    account_info: AccountInfo, batch_account_key: str, node_agent_sku_id: str
):
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
            and img.node_agent_sku_id == node_agent_sku_id
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

    # Find the network security group.
    network_security_groups = network_client.network_security_groups.list(
        account_info.resource_group
    )
    print(network_security_groups)

    network_security_group = None
    for nsg in network_security_groups:
        if nsg.name == vm_config["network_security_group"]:
            network_security_group = nsg
            break

    if network_security_group is None:
        print("Can't find network security group", vm_config["network_security_group"])
        return

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
            "network_security_group": network_security_group,
        },
    )

    nic_result = poller.result()

    print(f"Provisioned network interface client {nic_result.name}")

    # Provision the virtual machine
    compute_client = ComputeManagementClient(credential, account_info.subscription_id)

    vm_name = vm_config["name"]
    username = vm_config["admin_username"]
    ssh_username = vm_config["ssh_username"]
    ssh_user_publickey = vm_config["ssh_user_publickey"]
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
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": "/home/{}/.ssh/authorized_keys".format(
                                    ssh_username
                                ),
                                "key_data": ssh_user_publickey,
                            }
                        ]
                    },
                },
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


def create_pool(
    account_info: AccountInfo,
    credential: EnvironmentCredential,
    vm_name: str,
    vm_size: str,
    node_agent_sku_id: str,
    pool_config: dict(),
    gallery_config: dict(),
):
    compute_client = ComputeManagementClient(credential, account_info.subscription_id)

    # Find the VM
    all_vms = compute_client.virtual_machines.list(account_info.resource_group)

    vm = None
    for this_vm in all_vms:
        # print(this_vm)
        if this_vm.name == vm_name:
            vm = this_vm
            break

    if vm is None:
        print("Can't find vm", vm_name)
        return

    # Power off the VM
    poller = compute_client.virtual_machines.begin_power_off(
        account_info.resource_group, vm_name
    )
    power_off_result = poller.result()

    # Capture the VM
    vm_image_name = gallery_config["image_name"]
    compute_client.virtual_machines.generalize(account_info.resource_group, vm_name)

    source_sub_resource = SubResource(id=vm.id)
    image = Image(
        location=account_info.location, source_virtual_machine=source_sub_resource
    )
    poller = compute_client.images.begin_create_or_update(
        account_info.resource_group, vm_image_name, image
    )

    image_creation_result = poller.result()

    print(f"Created image {vm_image_name}. Result", image_creation_result)

    # Create the identifier for the image
    gallery_name = gallery_config["name"]
    gallery_image_name = gallery_config["gallery_image_name"]
    gallery_image_identifier = GalleryImageIdentifier(
        publisher=gallery_config["identifier"]["publisher"],
        offer=gallery_config["identifier"]["offer"],
        sku=gallery_config["identifier"]["sku"],
    )

    # If the image already exists in the gallery, delete it
    try:
        current_image = compute_client.gallery_images.get(
            account_info.resource_group, gallery_name, gallery_image_identifier
        )
        if current_image is not None:
            delete_poller = compute_client.gallery_images.begin_delete(
                account_info.resource_group, gallery_name, gallery_image_identifier
            )
            result = delete_poller.result()
            print(f"Deleted old image {gallery_image_identifier}")
    except azure.core.exceptions.ResourceNotFoundError:
        pass

    # Add the image to a gallery
    gallery_image = GalleryImage(
        location=account_info.location,
        description="test desc",
        os_type=OperatingSystemTypes.linux,
        os_state=OperatingSystemStateTypes.GENERALIZED,
        identifier=gallery_image_identifier,
    )
    create_image_poller = compute_client.gallery_images.begin_create_or_update(
        account_info.resource_group,
        gallery_name,
        gallery_image_name,
        gallery_image,
    )

    gallery_image_creation_result = create_image_poller.result()
    print(
        f"Created gallery image {gallery_image_name}. Result",
        gallery_image_creation_result,
    )

    # Add the image version to the image
    image_id = image_creation_result.id
    source = GalleryArtifactVersionSource(id=image_id)
    storage_profile = GalleryImageVersionStorageProfile(source=source)
    gallery_image_version = GalleryImageVersion(
        location=account_info.location, storage_profile=storage_profile
    )
    create_image_version_poller = (
        compute_client.gallery_image_versions.begin_create_or_update(
            account_info.resource_group,
            gallery_name,
            gallery_image_name,
            "1.0.0",
            gallery_image_version,
        )
    )

    image_version = create_image_version_poller.result()

    print("Added image to gallery", image_version)

    # Authenticate using the service principal
    client_id_var = "AZURE_CLIENT_ID"
    client_secret_var = "AZURE_CLIENT_SECRET"
    tenant_var = "AZURE_TENANT_ID"
    RESOURCE = "https://batch.core.windows.net/"
    client_id = os.getenv(client_id_var)
    if not client_id:
        raise Exception("Environment variable", client_id_var, "is not set.")

    client_secret = os.getenv(client_secret_var)
    if not client_secret:
        raise Exception("Environment variable", client_secret_var, "is not set.")

    tenant = os.getenv(tenant_var)
    if not tenant:
        raise Exception("Environment variable", tenant_var, "is not set.")

    credentials = ServicePrincipalCredentials(
        client_id=client_id,
        secret=client_secret,
        tenant=tenant,
        resource=RESOURCE,
    )

    # Create the pool
    batch_account_name = pool_config["batch_account_name"]
    batch_service_url = f"https://{batch_account_name}.westus3.batch.azure.com"

    batch_client = batch.BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    pool_id = pool_config["name"]
    pool_size = 5
    new_pool = batchmodels.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=batchmodels.ImageReference(virtual_machine_image_id=image_version.id),
            node_agent_sku_id=node_agent_sku_id,
        ),
        vm_size=vm_size,
        target_dedicated_nodes=pool_size,
    )
    pool_creation_result = batch_client.pool.add(new_pool)

    print(f"Created pool {pool_id}. Result", pool_creation_result)


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
    parser.add_argument(
        "--create-pool",
        dest="create_pool",
        action="store_const",
        const=True,
        default=False,
        help="Create the batch pool.",
    )

    args = parser.parse_args()

    print("Loading configuration from", args.configfile)
    with open(args.configfile) as f:
        config = json.load(f)

    account_info = AccountInfo(config)
    vm_name = config["vm_config"]["name"]
    vm_size = config["vm_size"]
    node_agent_sku_id = config["vm_config"]["node_agent_sku_id"]

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
        get_batch_vm_images(account_info, batch_account_key, node_agent_sku_id)
    if args.create_vm:
        provision_vm(account_info, credential, vm_size, config["vm_config"])
    if args.create_pool:
        create_pool(
            account_info,
            credential,
            vm_name,
            vm_size,
            node_agent_sku_id,
            config["pool_config"],
            config["gallery_config"],
        )


if __name__ == "__main__":
    # execute only if run as a script
    main()
