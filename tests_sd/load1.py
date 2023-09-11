import logging
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

from optimum.neuron import NeuronStableDiffusionPipeline

model_rootdir = '/home/ubuntu/sd_neuron'

kw = {
    "batch_size": 1, "height": 512, "width": 512, "device_ids": [0, 1],
    "dynamic_batching": False
}

stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_rootdir, **kw)

import pdb
pdb.set_trace()

