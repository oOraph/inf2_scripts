import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

from benchmark.utils import benchutils, loadutils  # noqa

args = benchutils.model_load_bench_init()

benchutils.test_load_time(args.measures, args.output, loadutils.load_neuron_diffusion,
                          'neuron load time', fn_kwargs=dict(model_id=args.model_id,
                                                             model_rootdir=args.model_dir))
