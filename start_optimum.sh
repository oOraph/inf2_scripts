#!/bin/bash

while true;do
    python /opt/test/benchmark/inference/optimum_neuron_inference.py -n 50 -s 42 -o /tmp/result1.csv -b 1 -m /shared/sd_neuron
done
