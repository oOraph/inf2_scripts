#!/bin/bash

set -e -x -o pipefail -u

echo '$nrconf{restart} = '"'a';" | sudo tee -a /etc/needrestart/needrestart.conf

echo Install Neuron drivers and tools

# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages 
sudo apt-get update -y

# Install OS headers 
sudo apt-get install linux-headers-$(uname -r) -y

# Install git 
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime 
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools 
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

echo Install docker

sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo mkdir -p /shared


# Model exported using the aws neuron sample notebook
# wget -O /tmp/sd2_compile_dir_512.tgz https://inf2-exports.s3.us-east-2.amazonaws.com/sd2_compile_dir_512.tgz
# sudo tar xf /tmp/sd2_compile_dir_512.tgz -C /shared
# rm -f /tmp/sd2_compile_dir_512.tgz
# This image is built with the provided Dockerfile.aws_export
# sudo docker run -d --device=/dev/neuron0 --name test1 -v /shared:/shared raphael31415/aws-inf2:1.0

# Model exported using a tweaked aws neuron sample notebook, removing the optimized attention
# wget -O /tmp/sd2_compile_dir_512_non_optimized.tgz https://inf2-exports.s3.us-east-2.amazonaws.com/sd2_compile_dir_512_non_optimized_attn.tgz
# sudo tar xf /tmp/sd2_compile_dir_512_non_optimized_attn.tgz -C /shared
# sudo mv /shared/sd2_compile_dir_512_non_optimized_attn /shared/sd2_compile_dir_512
# rm -f /tmp/sd2_compile_dir_512_non_optimized_attn.tgz
# This image is built with the provided Dockerfile.aws_export
# sudo docker run -d --device=/dev/neuron0 --name test1 -v /shared:/shared raphael31415/aws-inf2:1.0

# Model exported using optimum (without export NEURON_FUSE_SOFTMAX=1, e.g without optimized_attn)
# optimum-cli export neuron --model stabilityai/stable-diffusion-2-1-base \
#   --task stable-diffusion \
#   --batch_size 1 \
#   --height 512 `# height in pixels of generated image, eg. 512, 768` \
#   --width 512 `# width in pixels of generated image, eg. 512, 768` \
#   --auto-cast matmul \
#   --auto-cast-type bf16 \
#   sd_neuron/
# wget -O /tmp/sd_neuron.tgz https://inf2-exports.s3.us-east-2.amazonaws.com/sd_neuron.tgz
# sudo tar xf /tmp/sd_neuron.tgz -C /shared
# rm -f /tmp/sd_neuron.tgz
# This image is built with the provided Dockerfile.optimum_export
# sudo docker run -d --rm --device=/dev/neuron0 --name test1 -v /shared:/shared raphael31415/aws-inf2-optimum:1.0

# Model exported using optimum (with export NEURON_FUSE_SOFTMAX=1 beofre, e.g with optimized_attn)
# optimum-cli export neuron --model stabilityai/stable-diffusion-2-1-base \
#   --task stable-diffusion \
#   --batch_size 1 \
#   --height 512 `# height in pixels of generated image, eg. 512, 768` \
#   --width 512 `# width in pixels of generated image, eg. 512, 768` \
#   --auto-cast matmul \
#   --auto-cast-type bf16 \
#   sd_neuron/
wget -O /tmp/sd_neuron_optimized_attn.tgz https://inf2-exports.s3.us-east-2.amazonaws.com/sd_neuron_optimized_attn.tgz
sudo tar xf /tmp/sd_neuron_optimized_attn.tgz -C /shared
sudo mv /shared/sd_neuron_optimized_attn /shared/sd_neuron
rm -f /tmp/sd_neuron_optimized_attn.tgz
# This image is built with the provided Dockerfile.optimum_export
sudo docker run -d --rm --device=/dev/neuron0 --name test1 -v /shared:/shared raphael31415/aws-inf2-optimum:1.0

source ~/.bashrc

neuron-top
