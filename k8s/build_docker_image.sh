#!/bin/bash
# Build a Neuron container for running neuronx-nemo-megatron jobs on EKS
export DOCKER_BUILDKIT=1

# First build the Nemo and Apex wheels
./build.sh

# Specify AWS / ECR / repo info
AWS_ACCT=$(aws sts get-caller-identity | jq -r ".Account")
REGION=us-west-2
ECR_REPO=$AWS_ACCT.dkr.ecr.$REGION.amazonaws.com/neuronx_nemo

# Authenticate with ECR, build & push the image
aws ecr get-login-password --region $REGION | docker login --username AWS \
    --password-stdin $AWS_ACCT.dkr.ecr.$REGION.amazonaws.com \
  && docker build . -f ./k8s/docker/Dockerfile -t $ECR_REPO:latest \
  && docker push $ECR_REPO:latest
