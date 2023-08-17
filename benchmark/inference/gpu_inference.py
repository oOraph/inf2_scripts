from benchmark.utils import benchutils, loadutils

args, prompts = benchutils.latency_bench_init()
pipe = loadutils.load_gpu_diffusion(model=args.model_id)
benchutils.test_inference(args.output, pipe, prompts, args.batch_size, 'gpu latency')
