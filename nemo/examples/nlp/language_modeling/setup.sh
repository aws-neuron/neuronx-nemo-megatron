#! /bin/bash

set -e


# Configure Ubuntu for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
	    | sudo apt-key add -

# Update OS packages and install OS headers
sudo apt-get update -y && sudo apt-get install -y linux-headers-$(uname -r)

# Install Neuron OS packages and dependencies
sudo apt-get install -y aws-neuronx-dkms aws-neuronx-tools \
	    aws-neuronx-collectives=2.* aws-neuronx-runtime-lib=2.*

TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s  http://169.254.169.254/latest/meta-data/instance-id)
echo "instance_id:$INSTANCE_ID hostname:$(hostname)"
echo "runtime versions"
sudo apt list | grep neuron | grep installed

echo "compiler versions"
pip list | grep neuron

