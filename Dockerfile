# Based on installation instructions in https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md
# The following is the latest docker image as of this writing. Dependencies might need upgrades in future docker images.
# Supported docker images can be found in https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.12.0-ubuntu20.04

COPY . /workspace/nemo
RUN chmod 755 -R /workspace/nemo

WORKDIR /workspace/nemo

# Install Cython separately because Cython dependencies need to be in place for packages in requirements-docker.txt
RUN pip3 install wheel Cython

# Run the build script to create the neuronx-nemo-megatron wheels
RUN chmod 777 build.sh && ./build.sh

# Install the neuronx-nemo-megatron packages and dependencies
RUN pip3 install ./build/*.whl
RUN pip3 install -r requirements-docker.txt torch==1.13.1 protobuf==3.20.3

# Build the Megatron helper module. Ignore 'No neuron device available errors'
RUN cd ~ && python3 -c "from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper; \
compile_helper()"