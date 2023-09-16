This project "AWS Neuron Reference for NeMo Megatron" includes modified versions of the open-source packages [NeMo](https://github.com/NVIDIA/NeMo) and [Apex](https://github.com/NVIDIA/apex) that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances.

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

