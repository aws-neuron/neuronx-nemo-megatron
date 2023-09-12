This project "AWS Neuron Reference for NeMo Megatron" includes modified versions of the open-source packages [NeMo](https://github.com/NVIDIA/NeMo) and [Apex](https://github.com/NVIDIA/apex) that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances. Specifically we support 3D parallel model parallel sharding strategies for large language models. The APIs have been optimized for XLA based computation and high performance communication over Trainium instances. 

We support Temsor Parallel, Pipeline parallel and Data Parallel configurations for distributed training over a cluster of Tranium instances. We tested and added scripts for training 23B, 46B and 175B model configurations. To improve memory utilization we employ techniques such as sequence parallelism which reduces activation memory footprint, selective or full activation checkpointing which allows larger model configurations to fit. We also use SPMD optimizations whenever possible to reduce the number of graphs obtained. 

Please refer to the [neuronx-nemo-megatron GPT-3 pretraining tutorial](https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md) for instructions on how to use the code in this repository.

### Building inside docker

The following instructions have been verified to run on   
`763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.12.0-ubuntu20.04`

These instructions should be periodically updated for new docker images. Latest docker images can be found in https://github.com/aws/deep-learning-containers/blob/master/available_images.md

1. Authenticate for AWS ECR repository access to latest docker image
```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
```
2. Build the docker image
```bash
docker build -t neuron-nemo-megatron:dev .
```
3. You can now use this docker image to run neuron-nemo-megatron code, for e.g. 
```bash
cd /workspace/nemo/nemo/examples/nlp/language_modeling && ./test_llama.sh
```

