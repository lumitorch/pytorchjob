from typing import Optional, TypedDict, TypeVar, Any

import pulumi
import pulumi_kubernetes as kubernetes
from pulumi import ResourceOptions

__all__ = ["PyTorchJob", "PyTorchJobArgs"]

T = TypeVar("T")


# Normalize Input[T] to Output[T] and apply a default when the value is None
# Avoids using Python's `or`, which would clobber valid falsy values like 0 or "".
def with_default(value: Optional[pulumi.Input[T]], default: T) -> pulumi.Output[T]:
    return pulumi.Output.from_input(value).apply(lambda v: default if v is None else v)


# ---- Input validators / coercers -------------------------------------------
# Ensures we always have an Output[int] and fails fast with a helpful message
# if the user passes an invalid value (e.g., "four").


def _coerce_int(
    x: Any, *, name: str, min_: int | None = None, max_: int | None = None
) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(x, int):
        n = x
    elif isinstance(x, float) and x.is_integer():
        n = int(x)
    elif isinstance(x, str):
        s = x.strip()
        try:
            n = int(s, 10)
        except ValueError:
            raise TypeError(f"{name} must be an integer (got {x!r})")
    elif x is None:
        raise ValueError(f"{name} is required")
    else:
        raise TypeError(f"{name} must be an integer (got {type(x).__name__})")

    if min_ is not None and n < min_:
        raise ValueError(f"{name} must be ≥ {min_} (got {n})")
    if max_ is not None and n > max_:
        raise ValueError(f"{name} must be ≤ {max_} (got {n})")
    return n


def as_int(
    value: Optional[pulumi.Input[Any]],
    *,
    default: int | None,
    name: str,
    min_: int | None = None,
    max_: int | None = None,
) -> pulumi.Output[int]:
    # Normalize to Output, apply default if None, then validate/convert to int
    return pulumi.Output.from_input(value).apply(
        lambda v: _coerce_int(
            default if v is None else v, name=name, min_=min_, max_=max_
        )
    )


class PyTorchJobArgs(TypedDict):
    namespace: Optional[pulumi.Input[str]]
    """The namespace to create the PyTorchJob in. Defaults to `train`"""

    gpus_per_node: Optional[pulumi.Input[int]]
    """The number of GPUs per node. Defaults to `8`"""

    node_count: Optional[pulumi.Input[int]]
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

    def __init__(
        self, name: str, args: PyTorchJobArgs, opts: Optional[ResourceOptions] = None
    ) -> None:
        super().__init__("pytorch-job-component:index:PyTorchJob", name, {}, opts)

        if args.get("checkpoint_pvc_name") is None:
            raise ValueError("checkpoint_pvc_name is required")

        namespace = with_default(args.get("namespace"), "train")
        gpus_per_node = as_int(
            args.get("gpus_per_node"), default=8, name="gpus_per_node", min_=1
        )
        checkpoint_pvc_name = pulumi.Output.from_input(args.get("checkpoint_pvc_name"))
        pytorch_mnist_gpu_image_tag = with_default(
            args.get("pytorch_mnist_gpu_image_tag"), "v1beta1-8cd4b8c"
        )
        node_count = as_int(
            args.get("node_count"), default=2, name="node_count", min_=2
        )

        master_nodes = 1
        worker_nodes = node_count.apply(lambda n: max(n - master_nodes, 0))
        volume_mount_name = "checkpoint-storage"
        container_port = 23456

        # Ensure torchrun receives clean integer strings like "4", never "4.0"
        nnodes_str = node_count.apply(lambda n: str(n))
        nproc_str = gpus_per_node.apply(lambda g: str(g))

        common_env = [
            {"name": "NCCL_DEBUG", "value": "INFO"},
            {"name": "TORCH_NCCL_BLOCKING_WAIT", "value": "1"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
            {"name": "OMP_NUM_THREADS", "value": "1"},
        ]

        common_volume_mounts = [{"name": volume_mount_name, "mountPath": "/ckpt"}]

        torchrun_cmd = pulumi.Output.format(
            "torchrun --nnodes={0} --nproc_per_node={1} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
            "--no_python bash -lc 'export CUDA_VISIBLE_DEVICES=${{LOCAL_RANK}}; exec python /opt/pytorch-mnist/mnist.py --backend=nccl --epochs=100'",
            nnodes_str,
            nproc_str,
        )

        image_ref = pulumi.Output.format(
            "docker.io/kubeflowkatib/pytorch-mnist-gpu:{0}", pytorch_mnist_gpu_image_tag
        )

        pytorchJob = kubernetes.apiextensions.CustomResource(
            name,
            api_version="kubeflow.org/v1",
            kind="PyTorchJob",
            metadata={"name": name, "namespace": namespace},
            spec={
                "runPolicy": {"cleanPodPolicy": "None"},
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": master_nodes,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "runtimeClassName": "nvidia",
                                "volumes": [
                                    {
                                        "name": volume_mount_name,
                                        "persistentVolumeClaim": {
                                            "claimName": checkpoint_pvc_name
                                        },
                                    }
                                ],
                                "initContainers": [
                                    {
                                        "name": "warmup-dataset",
                                        "image": image_ref,
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": ["python", "-c"],
                                        "args": [
                                            """from torchvision.datasets import FashionMNIST
root="/ckpt/data"
# Download both splits once; subsequent ranks just read them.
FashionMNIST(root=root, train=True, download=True)
FashionMNIST(root=root, train=False, download=True)"""
                                        ],
                                        "volumeMounts": [
                                            {
                                                "name": volume_mount_name,
                                                "mountPath": "/ckpt",
                                            }
                                        ],
                                    }
                                ],
                                "containers": [
                                    {
                                        "name": "pytorch",
                                        "image": image_ref,
                                        "imagePullPolicy": "Always",
                                        "workingDir": "/ckpt",
                                        "env": common_env,
                                        "resources": {
                                            "requests": {
                                                "nvidia.com/gpu": gpus_per_node,
                                                "cpu": "2000m",
                                                "memory": "8Gi",
                                            },
                                            "limits": {
                                                "nvidia.com/gpu": gpus_per_node,
                                                "cpu": "4000m",
                                                "memory": "16Gi",
                                            },
                                        },
                                        "volumeMounts": common_volume_mounts,
                                        "command": ["bash", "-lc"],
                                        "args": [torchrun_cmd],
                                        "ports": [
                                            {
                                                "name": "pytorchjob-port",
                                                "containerPort": container_port,
                                            }
                                        ],
                                    }
                                ],
                            }
                        },
                    },
                    "Worker": {
                        "replicas": worker_nodes,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "runtimeClassName": "nvidia",
                                "volumes": [
                                    {
                                        "name": volume_mount_name,
                                        "persistentVolumeClaim": {
                                            "claimName": checkpoint_pvc_name
                                        },
                                    }
                                ],
                                "containers": [
                                    {
                                        "name": "pytorch",
                                        "image": image_ref,
                                        "imagePullPolicy": "Always",
                                        "workingDir": "/ckpt",
                                        "env": common_env,
                                        "resources": {
                                            "requests": {
                                                "nvidia.com/gpu": gpus_per_node,
                                                "cpu": "2000m",
                                                "memory": "8Gi",
                                            },
                                            "limits": {
                                                "nvidia.com/gpu": gpus_per_node,
                                                "cpu": "4000m",
                                                "memory": "16Gi",
                                            },
                                        },
                                        "volumeMounts": common_volume_mounts,
                                        "command": ["bash", "-lc"],
                                        "args": [torchrun_cmd],
                                        "ports": [
                                            {
                                                "name": "pytorchjob-port",
                                                "containerPort": container_port,
                                            }
                                        ],
                                    }
                                ],
                            }
                        },
                    },
                },
            },
            opts=pulumi.ResourceOptions(
                parent=self, provider=getattr(opts, "provider", None)
            ),
        )

        self.register_outputs(
            {
                "pytorchJob": pytorchJob,
                "namespace": namespace,
                "nodeCount": node_count,
                "gpusPerNode": gpus_per_node,
                "totalProcs": pulumi.Output.all(node_count, gpus_per_node).apply(
                    lambda xs: xs[0] * xs[1]
                ),
                "jobName": name,
            }
        )
