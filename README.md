This project "AWS Neuron Reference for NeMo Megatron" includes modified versions of the open-source packages [NeMo](https://github.com/NVIDIA/NeMo) and [Apex](https://github.com/NVIDIA/apex) that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances. Specifically we support 3D parallel model parallel sharding strategies for large language models. The APIs have been optimized for XLA based computation and high performance communication over Trainium instances. 

We support Temsor Parallel, Pipeline parallel and Data Parallel configurations for distributed training over a cluster of Tranium instances. We tested and added scripts for training 23B, 46B and 175B model configurations. To improve memory utilization we employ techniques such as sequence parallelism which reduces activation memory footprint, selective or full activation checkpointing which allows larger model configurations to fit. We also use SPMD optimizations whenever possible to reduce the number of graphs obtained. 

Please refer to the [neuronx-nemo-megatron GPT-3 pretraining tutorial](https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md) for instructions on how to use the code in this repository.
