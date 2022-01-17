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
import io
import time
import datetime

from azure.identity import AzureCliCredential, EnvironmentCredential
from azure.common.credentials import ServicePrincipalCredentials
from azure.mgmt import network

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

from azure.batch import BatchServiceClient
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels
import azure.core.exceptions

AUTOSCALE_FORMULA = """
    maxNumberOfVMs = {maxVMs};
    maxRecentTaskCount = max($PendingTasks.GetSample(30 * TimeInterval_Minute, 10));
    fullKeepAliveVMs = (maxRecentTaskCount > 0 ? $TargetDedicatedNodes : 0);
    keepAliveVMs = min(fullKeepAliveVMs, 1);
    curTaskCount = avg($PendingTasks.GetSample(1));
    curVMTarget = curTaskCount / $TaskSlotsPerNode + 0.51;
    newTargetNodes=min(maxNumberOfVMs, max(keepAliveVMs, curVMTarget));
    $TargetDedicatedNodes=newTargetNodes;
    $NodeDeallocationOption = taskcompletion;
    """


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


def get_batch_vm_images(account_info: AccountInfo, batch_account_key: str, node_agent_sku_id: str):
    """Queries Azure to find out which VM images can be used to create images for
    a batch pool.
    """
    batch_service_url = (
        f"https://{account_info.batch_account}.{account_info.location}.batch.azure.com"
    )

    credentials = batchauth.SharedKeyCredentials(account_info.batch_account, batch_account_key)

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Get VM images supported by Azure Batch
    options = batchmodels.AccountListSupportedImagesOptions(filter="verificationType eq 'verified'")
    images = list(
        batch_client.account.list_supported_images(account_list_supported_images_options=options)
    )
    filtered_images = list(
        filter(
            lambda img: img.os_type == batchmodels.OSType.linux
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
    """Provisions a VM using the given configuration."""
    # Code from https://docs.microsoft.com/en-us/azure/developer/python/azure-sdk-example-virtual-machines?tabs=cmd

    # Create a VM
    print("Provisioning a virtual machine...some operations might take a minute or two.")

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

    print(f"Provisioning virtual machine {vm_name}; this operation might take a few minutes.")

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
                                "path": "/home/{}/.ssh/authorized_keys".format(ssh_username),
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
    vnet_name: str,
    subnet_name: str,
    node_agent_sku_id: str,
    pool_config: dict(),
    gallery_config: dict(),
):
    """Creates an Azure Batch pool using the given VM."""
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
    poller = compute_client.virtual_machines.begin_power_off(account_info.resource_group, vm_name)
    power_off_result = poller.result()

    # Deallocate the VM so we don't get charged for it.
    poller = compute_client.virtual_machines.begin_deallocate(account_info.resource_group, vm_name)
    power_off_result = poller.result()

    # Capture the VM
    vm_image_name = gallery_config["image_name"]
    compute_client.virtual_machines.generalize(account_info.resource_group, vm_name)

    source_sub_resource = SubResource(id=vm.id)
    image = Image(location=account_info.location, source_virtual_machine=source_sub_resource)
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
    create_image_version_poller = compute_client.gallery_image_versions.begin_create_or_update(
        account_info.resource_group,
        gallery_name,
        gallery_image_name,
        "1.0.0",
        gallery_image_version,
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

    # Find the virtual network ID.
    network_client = NetworkManagementClient(credential, account_info.subscription_id)
    subnet_result = network_client.subnets.get(account_info.resource_group, vnet_name, subnet_name)

    # Create the pool
    batch_account_name = pool_config["batch_account_name"]
    batch_service_url = f"https://{batch_account_name}.westus3.batch.azure.com"

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    pool_id = pool_config["name"]
    autoscale_formula = AUTOSCALE_FORMULA.format(maxVMs=pool_config["maximum_nodes"])
    evaluation_interval = datetime.timedelta(minutes=5)
    network_config = batchmodels.NetworkConfiguration(subnet_id=subnet_result.id)
    public_ip_config = batchmodels.PublicIPAddressConfiguration(provision="noPublicIPAddresses")
    new_pool = batchmodels.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=batchmodels.ImageReference(virtual_machine_image_id=image_version.id),
            node_agent_sku_id=node_agent_sku_id,
        ),
        vm_size=vm_size,
        task_slots_per_node=pool_config["tasks_per_node"],
        task_scheduling_policy=batchmodels.TaskSchedulingPolicy(
            node_fill_type=batchmodels.ComputeNodeFillType.pack
        ),
        enable_auto_scale=True,
        auto_scale_formula=autoscale_formula,
        auto_scale_evaluation_interval=evaluation_interval,
        network_configuration=network_config,
        public_ip_address_configuration=public_ip_config,
    )
    pool_creation_result = batch_client.pool.add(new_pool)

    print(f"Created pool {pool_id}. Result", pool_creation_result)


def setup_pool_autoscale(
    account_info: AccountInfo,
    credential: EnvironmentCredential,
    pool_config: dict(),
):
    """Configures the pool autoscaling."""

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

    batch_account_name = pool_config["batch_account_name"]
    batch_service_url = f"https://{batch_account_name}.westus3.batch.azure.com"

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Update the pool
    pool_id = pool_config["name"]
    autoscale_formula = AUTOSCALE_FORMULA.format(maxVMs=pool_config["maximum_nodes"])
    evaluation_interval = datetime.timedelta(minutes=5)
    batch_client.pool.enable_auto_scale(pool_id, autoscale_formula, evaluation_interval)

    print(f"Updated autoscaling for pool {pool_id}.")


def remove_node(
    account_info: AccountInfo,
    credential: EnvironmentCredential,
    pool_config: dict(),
    node_id: str,
):
    """Removes a node from the pool."""

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

    batch_account_name = pool_config["batch_account_name"]
    batch_service_url = f"https://{batch_account_name}.westus3.batch.azure.com"

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Disable auto scaling
    pool_id = pool_config["name"]
    batch_client.pool.disable_auto_scale(pool_id)

    # Find the pool
    nodes = [node_id]
    remove_node_param = batchmodels.NodeRemoveParameter(
        node_list=nodes, node_deallocation_option="terminate"
    )
    batch_client.pool.remove_nodes(
        pool_id,
        remove_node_param,
    )

    print(f"Removed node {node_id} from {pool_id}.")

    # Renable auto scale
    setup_pool_autoscale(account_info, credential, pool_config)


def do_training(
    account_info: AccountInfo,
    batch_account_key: str,
    pool_config: dict,
    config_file: str,
):
    """Runs a one or more tasks to train an agents.
    The configuration file is a JSON file. If the top level is a dictionary, then
    it defines a single training. If it is an array, then each element defines
    a training.
    """

    batch_service_url = (
        f"https://{account_info.batch_account}.{account_info.location}.batch.azure.com"
    )

    credentials = batchauth.SharedKeyCredentials(account_info.batch_account, batch_account_key)

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Set up environment variables
    storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not storage_connection_string or len(storage_connection_string) < 5:
        print(
            "The AZURE_STORAGE_CONNECTION_STRING environment variable is not set or is too short."
        )

    env_vars = list()
    env_vars.append(
        batchmodels.EnvironmentSetting(
            name="AZURE_STORAGE_CONNECTION_STRING", value=storage_connection_string
        )
    )

    # Load the learning configuration
    with open(config_file, "r") as file:
        learning_config = json.load(file)

    if isinstance(learning_config, dict):
        # We want to run a single experiment
        learning_config = [learning_config]
    elif isinstance(learning_config, list):
        pass
    else:
        print("Unsupported type")
        return

    # Make a unique job ID
    timestr = datetime.datetime.now().isoformat()
    timestr = timestr.replace(":", "-")
    timestr = timestr.replace(".", "-")
    job_id = "Learning_" + timestr

    # Create a job
    pool_id = pool_config["name"]
    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        common_environment_settings=env_vars,
    )

    batch_client.job.add(job)

    # Create the tasks
    tasks = list()
    for learning, i in zip(learning_config, range(len(learning_config))):
        task_id = f"Task{i}"

        learning_config_json = json.dumps(learning)
        learning_config_json = learning_config_json.replace('"', '\\"')
        # print(learning_config_json)

        command = f"""/bin/bash -c "echo Learning task starting.;
        echo '{learning_config_json}' > config.json;
        echo Start config;
        cat config.json;
        echo End config;
        git clone https://github.com/davidnorman314/CEO.git;
        cd CEO/src;
        echo Starting python;
        python --version;
        source /home/david/py39/bin/activate;
        python -m learning.learning --azure ---post-train-stats-episodes 10000 --pickle-file ../../results.pkl ../../config.json;
        echo Python finished;
        echo Done;"
        """

        tasks.append(
            batchmodels.TaskAddParameter(
                id=task_id,
                command_line=command,
            )
        )

        batch_client.task.add_collection(job_id, tasks)

    # Wait for the job to complete
    while True:
        print("Checking tasks")
        job_tasks = list(batch_client.task.list(job_id))
        total = len(job_tasks)
        completed = 0
        for task in job_tasks:
            # print(task)

            if task.state == batchmodels.JobState.completed:
                completed += 1

        if completed == len(job_tasks):
            break

        print(f"Job {job_id} There are {completed} completed tasks of {total}")
        time.sleep(1)

    # Print information from the tasks
    job_tasks = batch_client.task.list(job_id)
    out_file_name = "stdout.txt"
    stderr_file_name = "stderr.txt"
    for task in job_tasks:
        node_id = batch_client.task.get(job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_client.file.get_from_task(job_id, task.id, out_file_name)

        file_text = _read_stream_as_string(stream)
        print("Standard output:")
        print(file_text)

        stream = batch_client.file.get_from_task(job_id, task.id, stderr_file_name)

        file_text = _read_stream_as_string(stream)
        print("Standard error:")
        print(file_text)


def run_test_job(
    account_info: AccountInfo,
    batch_account_key: str,
    pool_config: dict,
    task_count: int,
):
    """Runs a test job with the given number of tasks."""

    batch_service_url = (
        f"https://{account_info.batch_account}.{account_info.location}.batch.azure.com"
    )

    credentials = batchauth.SharedKeyCredentials(account_info.batch_account, batch_account_key)

    batch_client = BatchServiceClient(credentials, batch_service_url)
    batch_client.config.retry_policy.retries = 5

    # Set up environment variables
    env_vars = list()
    env_vars.append(batchmodels.EnvironmentSetting(name="TEST_ENV", value="abc"))

    # Make a unique job ID
    timestr = datetime.datetime.now().isoformat()
    timestr = timestr.replace(":", "-")
    timestr = timestr.replace(".", "-")
    job_id = "TestJob_" + timestr

    # Create a job
    pool_id = pool_config["name"]
    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        common_environment_settings=env_vars,
    )

    batch_client.job.add(job)

    # Add tasks
    tasks = list()
    for i in range(task_count):
        task_id = f"Task{i}"

        file = f"First line File{i}\nSecond line\nThird line."

        command = f"""/bin/bash -c "echo Task {i} executing.;
        echo Env var $TEST_ENV;
        CONFIG_VAR='{file}'
        echo $CONFIG_VAR > config.json
        git clone https://github.com/davidnorman314/CEO.git;
        ls CEO;
        ls CEO/src;
        echo Start file;
        cat config.json;
        echo End file;
        source /home/david/py39/bin/activate;
        python --version;
        echo Done;"
        """

        tasks.append(
            batchmodels.TaskAddParameter(
                id=task_id,
                command_line=command,
            )
        )

    batch_client.task.add_collection(job_id, tasks)

    # Wait for the job to complete
    while True:
        print("Checking tasks")
        job_tasks = list(batch_client.task.list(job_id))
        total = len(job_tasks)
        completed = 0
        for task in job_tasks:
            print(task)

            if task.state == batchmodels.JobState.completed:
                completed += 1

        if completed == len(job_tasks):
            break

        print(f"There are {completed} completed jobs of {total}")
        time.sleep(1)

    # Print information from the tasks
    job_tasks = batch_client.task.list(job_id)
    out_file_name = "stdout.txt"
    stderr_file_name = "stderr.txt"
    for task in job_tasks:
        node_id = batch_client.task.get(job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_client.file.get_from_task(job_id, task.id, out_file_name)

        file_text = _read_stream_as_string(stream)
        print("Standard output:")
        print(file_text)

        stream = batch_client.file.get_from_task(job_id, task.id, stderr_file_name)

        file_text = _read_stream_as_string(stream)
        print("Standard error:")
        print(file_text)


def _read_stream_as_string(stream):
    """
    Read stream as string

    :param stream: input stream generator
    :return: The file content.
    :rtype: str
    """
    encoding = "utf-8"

    output = io.BytesIO()
    try:
        for data in stream:
            output.write(data)
        return output.getvalue().decode(encoding)
    finally:
        output.close()


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
    parser.add_argument(
        "--setup-pool-autoscale",
        dest="setup_pool_autoscale",
        action="store_const",
        const=True,
        default=False,
        help="Set up the pool's autoscaling.",
    )
    parser.add_argument(
        "--remove-node",
        dest="remove_node",
        type=str,
        default=None,
        help="Removes a node from the pool.",
    )
    parser.add_argument(
        "--train",
        dest="learning_config",
        type=str,
        default=None,
        help="Executes one or more trainings as defined in the config file.",
    )
    parser.add_argument(
        "--run-test-job",
        dest="test_task_count",
        type=int,
        default=None,
        help="The number of tasks in the job.",
    )

    args = parser.parse_args()

    print("Loading configuration from", args.configfile)
    with open(args.configfile) as f:
        config = json.load(f)

    account_info = AccountInfo(config)
    vm_name = config["vm_config"]["name"]
    vm_size = config["vm_size"]
    vnet_name = config["vm_config"]["vnet_name"]
    subnet_name = config["vm_config"]["subnet_name"]
    node_agent_sku_id = config["vm_config"]["node_agent_sku_id"]
    pool_config = config["pool_config"]

    # Create a credential object from the environment.
    credential = EnvironmentCredential()

    batch_account_key = None
    if args.get_batch_vm_images or args.test_task_count or args.learning_config or args.remove_node:
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
            vnet_name,
            subnet_name,
            node_agent_sku_id,
            pool_config,
            config["gallery_config"],
        )
    if args.setup_pool_autoscale:
        setup_pool_autoscale(
            account_info,
            credential,
            pool_config,
        )
    if args.remove_node:
        remove_node(
            account_info,
            credential,
            pool_config,
            args.remove_node,
        )
    if args.test_task_count:
        run_test_job(account_info, batch_account_key, pool_config, args.test_task_count)
    if args.learning_config:
        do_training(account_info, batch_account_key, pool_config, args.learning_config)


if __name__ == "__main__":
    # execute only if run as a script
    main()
