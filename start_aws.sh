#!/bin/bash

while true; do
    python /opt/test/benchmark/inference/neuron_inference.py -n 50 -s 42 -o /tmp/result1.csv -b 1 -m /shared/sd2_compile_dir_512/
done
