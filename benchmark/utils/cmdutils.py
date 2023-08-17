import argparse


class ModelParserMixin(object):
    def __init__(self, *args, **kwargs):
        super(ModelParserMixin, self).__init__(*args, **kwargs)

        self.add_argument('-i', '--model-id',
                          default="stabilityai/stable-diffusion-2-1-base",
                          help="Model id",
                          dest="model_id")
        self.add_argument('-m', '--model-dir',
                          help='Directory containing the exported model',
                          default='./model_export',
                          dest='model_dir')


class BenchParserMixin(object):

    def __init__(self, *args, **kwargs):
        super(BenchParserMixin, self).__init__(*args, **kwargs)

        self.add_argument('-n', '--measures',
                          help='Number of times you the time measure will be performed',
                          default=10,
                          type=int,
                          dest='measures')
        self.add_argument('-b', '--batch-size', dest='batch_size',
                          help='Batch size', default=1, type=int)



class RandomParserMixin(object):

    def __init__(self, *args, **kwargs):
        super(RandomParserMixin, self).__init__(*args, **kwargs)
        self.add_argument('-s', '--seed',
                          help='Random seed to provide to random generator selecting prompts, '
                               'for reproducible benchmarks',
                          default=12,
                          type=int,
                          dest='seed')


class OutputParserMixin(object):

    def __init__(self, *args, **kwargs):
        super(OutputParserMixin, self).__init__(*args, **kwargs)

        self.add_argument('-o', '--output', dest='output',
                          help='Bench result output file', default='/tmp/result.csv')


class BenchParser(ModelParserMixin, BenchParserMixin, OutputParserMixin, RandomParserMixin, argparse.ArgumentParser):
    pass


class LoadParser(ModelParserMixin, BenchParserMixin, OutputParserMixin, argparse.ArgumentParser):
    pass


def parse_inference_bench_cmdline(argv=None):
    parser = BenchParser()
    return parser.parse_args(argv)


def parse_load_bench_cmdline(argv=None):
    parser = LoadParser()
    return parser.parse_args(argv)
