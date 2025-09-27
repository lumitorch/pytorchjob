# PyTorchJob Pulumi Component

A Pulumi component for deploying distributed PyTorch training jobs on Kubernetes using the Kubeflow PyTorchJob Custom Resource.

## Overview

This component provides a simple way to deploy distributed PyTorch training jobs on your Kubernetes cluster with GPU support. The component creates a PyTorchJob Custom Resource that runs a Fashion MNIST training job using PyTorch's torchrun utility for distributed training across multiple GPU-enabled nodes.

## Features

- **Distributed PyTorch Training**: Supports multi-node, multi-GPU distributed training
- **Fashion MNIST Example**: Pre-configured Fashion MNIST training job
- **GPU Acceleration**: Full NVIDIA GPU support with configurable GPU allocation
- **Checkpoint Storage**: Persistent volume support for model checkpoints and dataset storage
- **Master-Worker Architecture**: Automatically configures master and worker nodes
- **Dataset Initialization**: Pre-downloads datasets before training begins
- **Configurable Resources**: Flexible GPU and node count configuration

## Architecture

The component creates the following resources:

1. **PyTorchJob Custom Resource**: Main training job definition using Kubeflow's PyTorchJob CRD
2. **Master Node**: Single master node responsible for coordination and training
3. **Worker Nodes**: Configurable number of worker nodes for distributed training
4. **Init Container**: Downloads Fashion MNIST dataset before training starts
5. **Persistent Volume**: Shared storage for checkpoints and datasets

## Requirements

- Kubernetes cluster with Kubeflow Training Operator installed
- GPU-enabled nodes with NVIDIA drivers
- Persistent volume for checkpoint storage
- Pulumi CLI
- Python 3.7+

## Installation

### Prerequisites

Ensure you have the required dependencies:

```bash
pip install pulumi>=3.0.0,<4.0.0 pulumi-kubernetes>=4.0.0,<5.0.0
```

### Usage

```python
from pytorchjob import PyTorchJob

# Basic usage with default settings
pytorch_job = PyTorchJob("my-pytorch-job", {
    "checkpoint_pvc_name": "my-checkpoint-pvc"  # Required
})

# Custom configuration
pytorch_job = PyTorchJob("distributed-training", {
    "namespace": "ml-training",                    # Optional: defaults to "train"
    "gpus_per_node": 4,                           # Optional: defaults to 8
    "node_count": 4,                              # Optional: defaults to 2
    "checkpoint_pvc_name": "training-storage",    # Required
    "pytorch_mnist_gpu_image_tag": "latest"       # Optional: defaults to "v1beta1-8cd4b8c"
})
```

## Configuration

### PyTorchJobArgs

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `checkpoint_pvc_name` | `str` | Yes | - | Name of the PersistentVolumeClaim for checkpoint storage |
| `namespace` | `str` | No | `train` | Kubernetes namespace to deploy the job |
| `gpus_per_node` | `int` | No | `8` | Number of GPUs per node |
| `node_count` | `int` | No | `2` | Total number of nodes (minimum 2) |
| `pytorch_mnist_gpu_image_tag` | `str` | No | `v1beta1-8cd4b8c` | Docker image tag for PyTorch MNIST |

### Configuration Notes

- **Minimum Node Count**: The component requires at least 2 nodes (1 master + 1 worker)
- **GPU Requirements**: Each node should have the specified number of GPUs available
- **Runtime Class**: Uses `nvidia` runtime class for GPU access
- **Network Configuration**: Uses NCCL backend for distributed communication

## Training Job Details

### Dataset
- **Fashion MNIST**: A dataset of clothing item images (28x28 grayscale)
- **Automatic Download**: Dataset is downloaded during initialization
- **Shared Storage**: Dataset stored on persistent volume for all nodes

### Distributed Training
- **torchrun**: Uses PyTorch's native distributed launcher
- **NCCL Backend**: Optimized for GPU-to-GPU communication
- **Multi-Node**: Supports scaling across multiple nodes
- **Multi-GPU**: Supports multiple GPUs per node

### Environment Variables
- `NCCL_DEBUG`: Set to INFO for debugging communication
- `TORCH_NCCL_BLOCKING_WAIT`: Ensures synchronous operations
- `NCCL_SOCKET_IFNAME`: Network interface for NCCL (eth0)
- `OMP_NUM_THREADS`: OpenMP thread count (set to 1)

## Example: Complete Deployment

```python
import pulumi
import pulumi_kubernetes as k8s
from pytorchjob import PyTorchJob

# Create namespace for training
train_namespace = k8s.core.v1.Namespace(
    "train-namespace",
    metadata=k8s.meta.v1.ObjectMetaArgs(name="train")
)

# Create persistent volume claim for checkpoints
checkpoint_pvc = k8s.core.v1.PersistentVolumeClaim(
    "checkpoint-storage",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="pytorch-checkpoint-pvc",
        namespace="train"
    ),
    spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
        access_modes=["ReadWriteMany"],
        resources=k8s.core.v1.ResourceRequirementsArgs(
            requests={"storage": "100Gi"}
        )
    ),
    opts=pulumi.ResourceOptions(depends_on=[train_namespace])
)

# Deploy PyTorch training job
pytorch_job = PyTorchJob("fashion-mnist-training", {
    "namespace": "train",
    "gpus_per_node": 8,
    "node_count": 2,
    "checkpoint_pvc_name": "pytorch-checkpoint-pvc",
    "pytorch_mnist_gpu_image_tag": "v1beta1-8cd4b8c"
}, opts=pulumi.ResourceOptions(depends_on=[checkpoint_pvc]))
```

## Monitoring and Management

Monitor your PyTorch job using `kubectl`:

```bash
# Check PyTorchJob status
kubectl get pytorchjobs -n train

# Check job pods
kubectl get pods -n train -l job-name=pytorch-mnist-dist

# View training logs (master node)
kubectl logs -n train -l pytorch-job-role=master -f

# View training logs (worker nodes)
kubectl logs -n train -l pytorch-job-role=worker -f

# Check persistent volume usage
kubectl get pvc -n train
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check GPU availability and node labels
2. **NCCL communication errors**: Verify network connectivity between nodes
3. **Storage issues**: Ensure PVC is properly mounted and accessible
4. **Image pull failures**: Verify Docker image availability and registry access

### GPU Requirements
- Nodes must have NVIDIA GPUs with proper drivers
- GPU runtime must be configured (nvidia-docker2 or containerd)
- Nodes should be labeled appropriately for GPU workloads

### Network Requirements
- Inter-node communication must be enabled
- Port 23456 is used for PyTorch distributed communication
- NCCL requires low-latency networking for optimal performance

### Verification Commands

```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-v100

# Verify PyTorchJob CRD
kubectl get crd pytorchjobs.kubeflow.org

# Check Kubeflow Training Operator
kubectl get pods -n kubeflow
```

## Version Compatibility

- PyTorch: Compatible with PyTorch 1.9+ (via Docker image)
- Kubernetes: 1.20+
- Kubeflow Training Operator: 1.3+
- Pulumi: 3.0+
- Pulumi Kubernetes Provider: 4.0+
- NVIDIA Driver: 470.57.02+
- CUDA: 11.4+

## Performance Considerations

- **Storage**: Use high-performance storage (NVMe SSD) for checkpoints
- **Networking**: InfiniBand or 100GbE recommended for multi-node training
- **GPU Memory**: Ensure sufficient GPU memory for batch size and model
- **Node Resources**: Balance CPU, memory, and GPU allocation

## Contributing

This component is part of the LumiTorch infrastructure toolkit. For issues and contributions, please refer to the project repository.

## License

Please refer to the project's license file for licensing information.
