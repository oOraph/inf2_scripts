from benchmark.utils import benchutils, loadutils

args = benchutils.model_load_bench_init()

benchutils.test_load_time(args.measures, args.output, loadutils.load_gpu_diffusion,
                          'gpu load time', fn_kwargs=dict(model=args.model_id))
