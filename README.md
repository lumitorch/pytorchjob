PyTorchJob Pulumi Component

  A Pulumi component for deploying distributed PyTorch training jobs on
  Kubernetes using Kubeflow Training Operator's PyTorchJob CRD. Supports
  multi-node FSDP training with NCCL communication and Kueue queue integration.

  Key Features

  - Distributed multi-node, multi-GPU PyTorch training
  - NCCL communication for efficient distributed training
  - FSDP (Fully Sharded Data Parallel) support
  - Kueue integration for queue management and resource scheduling
  - Pre-configured Fashion MNIST training job
  - Full NVIDIA GPU support
  - Persistent volume for checkpoints and datasets
  - Automatic master-worker node configuration
  - Dataset pre-download capability
  - Flexible GPU and node resource configuration

  Architecture

  The component creates:
  1. PyTorchJob Custom Resource
  2. Master Node for coordination
  3. Configurable Worker Nodes
  4. Init Container for dataset download
  5. Persistent Volume for storage

  Component Integration

  Works seamlessly with other LumiTorch components:
  - Kubeflow Training Operator: Provides PyTorchJob CRD (required)
  - Kueue: Queue management for training workloads (optional)
  - GPU Operator: GPU resource management and monitoring (recommended)

  Requirements

  - Kubernetes cluster with Kubeflow Training Operator
  - GPU-enabled nodes with NVIDIA drivers
  - Persistent volume storage
  - Pulumi CLI
  - Python 3.7+

  Basic Usage

  from pytorchjob import PyTorchJob

  # Basic configuration
  pytorch_job = PyTorchJob("my-pytorch-job", {
      "checkpoint_pvc_name": "my-checkpoint-pvc"
  })

  # Advanced distributed training configuration
  pytorch_job = PyTorchJob("distributed-training", {
      "namespace": "train",
      "nnodes": 2,
      "nproc_per_node": 8,
      "gpus_per_node": 4,
      "checkpoint_pvc_name": "training-storage",
      "pytorch_mnist_gpu_image_tag": "latest"
  })

  Distributed Training Configuration

  NCCL Environment Variables

  pytorch_job = PyTorchJob("nccl-training", {
      "environment_vars": {
          "NCCL_DEBUG": "INFO",
          "TORCH_NCCL_BLOCKING_WAIT": "1",
          "NCCL_SOCKET_IFNAME": "eth0"
      },
      "fsdp_enabled": True,
      "nnodes": 2,
      "nproc_per_node": 8
  })

  Multi-Node Setup

  pytorch_job = PyTorchJob("multi-node-training", {
      "namespace": "train",
      "nnodes": 4,  # 1 master + 3 workers
      "nproc_per_node": 8,
      "gpus_per_node": 1,
      "checkpoint_pvc_name": "shared-storage",
      "requests_equal_limits": True  # Set requests = limits for GPUs
  })

  Queue Management Integration

  Integrate with Kueue for resource scheduling:

  pytorch_job = PyTorchJob("queued-training", {
      "namespace": "train",  # Must match LocalQueue namespace
      "queue_name": "training-queue",
      "cluster_queue": "gpu-cluster-queue",
      "suspend_job": True  # Let Kueue control scheduling
  })

  Configuration Parameters

  | Parameter                   | Type | Required | Default  | Description
                      |
  |-----------------------------|------|----------|----------|------------------
  --------------------|
  | checkpoint_pvc_name         | str  | Yes      | -        |
  PersistentVolumeClaim name           |
  | namespace                   | str  | No       | "train"  | Kubernetes
  namespace                 |
  | nnodes                      | int  | No       | 2        | Total nodes
  (master + workers)       |
  | nproc_per_node              | int  | No       | 8        | Processes per
  node                   |
  | gpus_per_node               | int  | No       | 8        | GPUs per node
                      |
  | node_count                  | int  | No       | 2        | Total nodes
  (deprecated, use nnodes) |
  | pytorch_mnist_gpu_image_tag | str  | No       | "latest" | Container image
  tag                  |
  | environment_vars            | dict | No       | {}       | Environment
  variables                |
  | fsdp_enabled                | bool | No       | False    | Enable FSDP
  training                 |
  | queue_name                  | str  | No       | None     | Kueue LocalQueue
  name                |
  | cluster_queue               | str  | No       | None     | Kueue
  ClusterQueue name              |
  | suspend_job                 | bool | No       | False    | Start job in
  suspended state         |
  | requests_equal_limits       | bool | No       | True     | Set resource
  requests = limits       |

  Smoke Testing and Verification

  Deploy a smoke test PyTorchJob:

  smoke_test = PyTorchJob("pytorch-smoke-test", {
      "namespace": "train",
      "nnodes": 2,  # 1 master + 1 worker
      "nproc_per_node": 8,
      "gpus_per_node": 1,
      "checkpoint_pvc_name": "test-storage",
      "environment_vars": {
          "NCCL_DEBUG": "INFO",
          "TORCH_NCCL_BLOCKING_WAIT": "1",
          "NCCL_SOCKET_IFNAME": "eth0"
      }
  })

  # Export verification outputs
  pulumi.export("pytorch_job_status", smoke_test.status)
  pulumi.export("checkpoint_location", "/ckpt")

  Monitoring and Verification

  Monitor training jobs using kubectl:

  # Check PyTorchJob status
  kubectl get pytorchjobs -n train

  # View job details
  kubectl describe pytorchjob my-pytorch-job -n train

  # Check worker pods
  kubectl get pods -l job-name=my-pytorch-job -n train

  # View training logs
  kubectl logs -l job-name=my-pytorch-job,replica-type=master -n train

  # Verify checkpoints
  kubectl exec -it <worker-pod> -- ls -la /ckpt

  Resource Management

  - Set requests = limits for GPU resources to ensure proper scheduling
  - Schedule one worker per node to preserve GPU locality
  - Monitor GPU utilization via DCGM metrics (requires GPU Operator)
  - Use node taints and tolerations for GPU scheduling

  Troubleshooting

  Common issues:
  - Verify Kubeflow Training Operator is installed and running
  - Check GPU node labels and taints match job requirements
  - Confirm persistent volume is accessible from worker nodes
  - Validate NCCL communication between nodes
  - Ensure sufficient GPU resources in ClusterQueue (if using Kueue)

  Contributing

  Part of the LumiTorch infrastructure toolkit. Contributions welcome via GitHub
   issues and pull requests.
