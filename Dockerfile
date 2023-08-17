FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.12.0-ubuntu20.04

RUN mkdir -p /opt/test

COPY benchmark/ /opt/test/

ENTRYPOINT ["/bin/bash"]

ENV PYTHONPATH "/opt/test"
ENV NEURON_FUSE_SOFTMAX "1"

RUN pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
RUN pip install accelerate optimum[neuronx] diffusers

ENTRYPOINT ["python", "/opt/test/benchmark/inference/optimum_neuron_inference.py", "-n", "50", "-s", "42", "-o", "/tmp/result1.csv", "-b", "1", "-m", "/shared/sd_neuron/"]
