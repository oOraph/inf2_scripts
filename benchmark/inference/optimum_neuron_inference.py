import logging
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

from optimum.neuron import NeuronStableDiffusionPipeline

from benchmark.utils import benchutils, timeutils  # noqa

args, prompts = benchutils.latency_bench_init()

LOG = logging.getLogger(__name__)


@timeutils.timeit
def load_optimum_neuron_diffusion(model_rootdir, dynamic_batching=False):

    kw = {
        "batch_size": 1, "height": 512, "width": 512, "device_ids": [0, 1],
        "dynamic_batching": dynamic_batching
    }

    stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_rootdir, **kw)

    LOG.debug("Optimum model loaded")

    return stable_diffusion


pipe = load_optimum_neuron_diffusion(model_rootdir=args.model_dir,
                                     dynamic_batching=args.batch_size > 1)

benchutils.test_inference(args.output, pipe, prompts, args.batch_size, 'neuron latency')
