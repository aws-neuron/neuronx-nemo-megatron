This directory contains a Dockerfile and related assets to enable neuronx-nemo-megatron 
distributed training jobs on EKS.

## Prereqs
To use this code, it is assumed that you already have an EKS cluster with a trn1 nodegroup, 
Neuron/EFA plugins, and FSx storage as outlined in [this tutorial](https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain).
**Note:** torchx and volcano are mentioned in the tutorial, but are not required to run neuronx-nemo-megatron on EKS.

Additionally, you will need to install the [MPI Operator for Kubernetes](https://github.com/kubeflow/mpi-operator) 
on your cluster.

To run the included GPT pretraining examples, download the preprocessed training dataset
to the FSx storage attached to your EKS cluster as follows:
* First, launch and then connect to a temporary pod that has access to the FSx shared volume. Here we assume the volume is located at `/shared` within the pod.
* Within the temporary pod, run the following commands to download the GPT training data to /shared:
```
mkdir -p /shared/examples_datasets/gpt2
cd /shared/examples_datasets/gpt2
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.bin .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.idx .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/license.txt .  --no-sign-request
```

## Build the neuronx-nemo-megatron container image
The following steps can be run on a cloud desktop or vanilla EC2 linux instance (trn1 not required).

First make sure that your Python environment has the wheel and torch packages
```
pip3 install wheel torch
```

Copy/clone the contents of this repository to your cloud desktop / linux instance.

Create a new ECR repository (ex: neuronx_nemo) in the AWS region you intend to use.

Modify `build_docker_image.sh` to specify the correct AWS region and ECR repo.

Ensure that you have configured AWS credentials on your instance with permission to login
and push images to ECR.

From the root of the cloned repository, run `./k8s/build_docker_image.sh` to build
the Nemo/Apex packages, build the neuronx-nemo-megatron container image, and then
push the image to your ECR repo.

## Launch neuronx-nemo-megatron GPT pretraining job on EKS
Copy the contents of the `example_manifests` directory to the instance you use to 
manage your EKS cluster.

Modify the contents of the example manifests to reference your ECR repo / image on
the various lines containing the `image:` definitions.

Launch the ahead-of-time compilation job:
```
kubectl apply -f ./mpi_compile.yaml
```

Once the compilation job is complete, launch the training job:
```
kubectl apply -f ./mpi_train.yaml
```

