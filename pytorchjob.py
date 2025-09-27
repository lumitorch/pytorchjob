from typing import Optional, TypedDict

import pulumi
import pulumi_kubernetes as kubernetes
from pulumi import ResourceOptions


class PyTorchJobArgs(TypedDict):
    namespace: Optional[pulumi.Input[str]]
    """The namespace to create the PyTorchJob in. Defaults to `train`"""

    gpus_per_node: pulumi.Input[int]
    """The number of GPUs per node. Defaults to `8`"""

    node_count: pulumi.Input[int]
    """The number of nodes in the cluster. Defaults to `2`"""

    checkpoint_pvc_name: pulumi.Input[str]
    """The name of the PersistentVolumeClaim used for checkpoint storage"""

    pytorch_mnist_gpu_image_tag: Optional[pulumi.Input[str]]
    """The docker image tag to use for pytorch MNIST. Defaults to `v1beta1-8cd4b8c`"""


class PyTorchJob(pulumi.ComponentResource):
    """
    Defines a Pulumi ComponentResource for deploying a distributed PyTorch training job
    on a Kubernetes cluster using the Kubeflow PyTorchJob Custom Resource. The training job
    expects at least two worker nodes with at least one GPU each.

    This component sets up the necessary configuration and resources to run a distributed
    Fashion MNIST training job using PyTorch with GPU support. It includes a master node
    and a worker node, with support for initializing the dataset prior to training. The
    job uses PyTorch's torchrun utility for distributed training and provides the ability
    to specify custom resource configurations such as the number of GPUs per node.
    """

    def __init__(self,
                 name: str,
                 args: PyTorchJobArgs,
                 opts: Optional[ResourceOptions] = None) -> None:
        super().__init__('pytorch-job-component:index:PyTorchJob', name, {}, opts)

        namespace = args.get("namespace", "train")
        gpus_per_node = args.get("gpus_per_node", 8)
        checkpoint_pvc_name = args.get("checkpoint_pvc_name")
        pytorch_mnist_gpu_image_tag = args.get("pytorch_mnist_gpu_image_tag", "v1beta1-8cd4b8c")
        node_count = args.get("node_count", 2)

        assert node_count >= 2, "Node count must be at least 2"
        master_nodes = 1
        worker_nodes = node_count - master_nodes

        volume_mount_name = "checkpoint-storage"
        container_port = 23456

        pytorchJob = kubernetes.apiextensions.CustomResource(
            "pytorch-job",
            api_version="kubeflow.org/v1",
            kind="PyTorchJob",
            metadata={
                "name": "pytorch-mnist-dist",
                "namespace": namespace
            },
            spec={
                "runPolicy": {
                    "cleanPodPolicy": "None"
                },
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": master_nodes,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "runtimeClassName": "nvidia",
                                "volumes": [{
                                    "name": volume_mount_name,
                                    "persistentVolumeClaim": {
                                        "claimName": checkpoint_pvc_name
                                    }
                                }],
                                "initContainers": [{
                                    "name": "warmup-dataset",
                                    "image": f"docker.io/kubeflowkatib/pytorch-mnist-gpu:{pytorch_mnist_gpu_image_tag}",
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["python", "-c"],
                                    "args": [
                                        '''from torchvision.datasets import FashionMNIST
root="/ckpt/data"
# Download both splits once; subsequent ranks just read them.
FashionMNIST(root=root, train=True, download=True)
FashionMNIST(root=root, train=False, download=True)'''
                                    ],
                                    "volumeMounts": [{
                                        "name": volume_mount_name,
                                        "mountPath": "/ckpt"
                                    }]
                                }],
                                "containers": [{
                                    "name": "pytorch",
                                    "image": f"docker.io/kubeflowkatib/pytorch-mnist-gpu:{pytorch_mnist_gpu_image_tag}",
                                    "imagePullPolicy": "Always",
                                    "workingDir": "/ckpt",
                                    "env": [
                                        {"name": "NCCL_DEBUG", "value": "INFO"},
                                        {"name": "TORCH_NCCL_BLOCKING_WAIT", "value": "1"},
                                        {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                                        {"name": "OMP_NUM_THREADS", "value": "1"}
                                    ],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": gpus_per_node},
                                        "limits": {"nvidia.com/gpu": gpus_per_node}
                                    },
                                    "volumeMounts": [{
                                        "name": volume_mount_name,
                                        "mountPath": "/ckpt"
                                    }],
                                    "command": ["bash", "-lc"],
                                    "args": [
                                        f"torchrun --nnodes={node_count} --nproc_per_node={gpus_per_node} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --no_python bash -lc 'export CUDA_VISIBLE_DEVICES=${{LOCAL_RANK}}; exec python /opt/pytorch-mnist/mnist.py --backend=nccl'"],
                                    "ports": [{
                                        "name": "pytorchjob-port",
                                        "containerPort": container_port
                                    }]
                                }]
                            }
                        }
                    },
                    "Worker": {
                        "replicas": worker_nodes,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "runtimeClassName": "nvidia",
                                "volumes": [{
                                    "name": volume_mount_name,
                                    "persistentVolumeClaim": {
                                        "claimName": checkpoint_pvc_name
                                    }
                                }],
                                "containers": [{
                                    "name": "pytorch",
                                    "image": f"docker.io/kubeflowkatib/pytorch-mnist-gpu:{pytorch_mnist_gpu_image_tag}",
                                    "imagePullPolicy": "Always",
                                    "workingDir": "/ckpt",
                                    "env": [
                                        {"name": "NCCL_DEBUG", "value": "INFO"},
                                        {"name": "TORCH_NCCL_BLOCKING_WAIT", "value": "1"},
                                        {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                                        {"name": "OMP_NUM_THREADS", "value": "1"}
                                    ],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": gpus_per_node},
                                        "limits": {"nvidia.com/gpu": gpus_per_node}
                                    },
                                    "volumeMounts": [{
                                        "name": volume_mount_name,
                                        "mountPath": "/ckpt"
                                    }],
                                    "command": ["bash", "-lc"],
                                    "args": [
                                        f"torchrun --nnodes={node_count} --nproc_per_node={gpus_per_node} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --no_python bash -lc 'export CUDA_VISIBLE_DEVICES=${{LOCAL_RANK}}; exec python /opt/pytorch-mnist/mnist.py --backend=nccl'"],
                                    "ports": [{
                                        "name": "pytorchjob-port",
                                        "containerPort": container_port
                                    }]
                                }]
                            }
                        }
                    }
                }
            },
            opts=pulumi.ResourceOptions(parent=self, provider=opts.provider)
        )

        self.register_outputs({})
