import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

from benchmark.utils import benchutils, loadutils  # noqa

args, prompts = benchutils.latency_bench_init()
pipe = loadutils.load_neuron_diffusion(model_id=args.model_id,
                                       model_rootdir=args.model_dir,
                                       dynamic_batching=args.batch_size > 1)
benchutils.test_inference(args.output, pipe, prompts, args.batch_size, 'neuron latency')
