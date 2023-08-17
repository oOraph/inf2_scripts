import logging
import time

import numpy as np
import pandas as pd


from benchmark.utils import cmdutils, loadutils, randomutils

LOG = logging.getLogger(__name__)


def _common_inference_bench_init(cmdline_fn):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
                        force=True)
    args = cmdline_fn()
    randomutils.init_with_seed(args.seed)
    prompts = loadutils.load_prompts(args.measures * args.batch_size)
    # Check the specified output is writable, avoids bad surprises at the end of the bench
    with open(args.output, 'w') as f:
        pass
    return args, prompts


def latency_bench_init():
    return _common_inference_bench_init(cmdutils.parse_inference_bench_cmdline)


def model_load_bench_init():
    logging.basicConfig(level=logging.DEBUG, force=True)
    args = cmdutils.parse_load_bench_cmdline()
    # Check the specified output is writable, avoids bad surprises at the end of the bench
    with open(args.output, 'w') as f:
        pass
    return args


def _test_inference_common(bench_dst, model, prompts, batch_size, bench_name=None):
    measures = []
    total = len(prompts)
    count = 0
    for i in range(0, total, batch_size):
        x = prompts[i:i+batch_size]['Prompt']
        if len(x) != batch_size:
            continue
        start = time.time()
        model(x)
        elapsed = time.time() - start
        measures.append(elapsed)
        count += 1
        LOG.info("Progress %d / %d", i + batch_size, total)
    # TODO: compute CLIP and/or FID score, to verify the quality is as good as the original model
    results = basic_stats(measures, bench_dst,
                          batch_size=batch_size, count=count,
                          bench=bench_name)
    return results


def test_inference(bench_dst, model, prompts, batch_size=1, bench_name=None):
    LOG.info("Testing model inference")
    return _test_inference_common(bench_dst, model, prompts, batch_size, bench_name)


def test_load_time(measures, output_file, load_func, bench_name=None, fn_args=(), fn_kwargs={}):
    t_measures = []
    for i in range(0, measures):
        LOG.info("Measuring model load time %d / %d", i, measures)
        r = load_func(*fn_args, **fn_kwargs)
        t_measures.append(r.elapsed_time)
    results = basic_stats(t_measures, output_file,
                          count=measures,
                          bench=bench_name)
    return results


def basic_stats(row, output, **data):
    if not isinstance(row, np.ndarray):
        row = np.array(row)
    avg = row.mean(axis=0)
    std = row.std(axis=0)
    median = np.median(row)
    p95 = np.quantile(row, 0.95, axis=0)
    result = pd.DataFrame([[avg, std, median, p95]],
                          columns=['avg', 'std', 'p50', 'p95'])
    for k, v in data.items():
        result[k] = v
    result.to_csv(output, index=False)
    LOG.info("Stats %s", result.head())
    return result
