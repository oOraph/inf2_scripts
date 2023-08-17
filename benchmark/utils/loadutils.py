# Mainly adapted from
# https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_512_inference.ipynb

import logging
import os

import datasets
import numpy as np
import torch
import torch.nn as nn
try:
    import torch_neuronx  # noqa
except:
    pass
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from benchmark.utils import timeutils


LOG = logging.getLogger(__name__)


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.to(dtype=torch.bfloat16).expand((sample.shape[0],)),
                               encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


@timeutils.timeit
def load_neuron_diffusion(model_id, model_rootdir, dynamic_batching=False):

    LOG.info("Loading neuron model for %s, from %s", model_id, model_rootdir)

    text_encoder_filename = os.path.join(model_rootdir, 'text_encoder/model.pt')
    decoder_filename = os.path.join(model_rootdir, 'vae_decoder/model.pt')
    unet_filename = os.path.join(model_rootdir, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(model_rootdir, 'vae_post_quant_conv/model.pt')

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Replaces StableDiffusionPipeline's decode_latents method with our custom decode_latents method defined above.
    StableDiffusionPipeline.decode_latents = decode_latents

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0, 1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids,
                                                    set_dynamic_batching=dynamic_batching)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

    if dynamic_batching:
        pipe.text_encoder.neuron_text_encoder = \
            torch_neuronx.dynamic_batch(pipe.text_encoder.neuron_text_encoder)
        pipe.vae.decoder = torch_neuronx.dynamic_batch(pipe.vae.decoder)
        pipe.vae.post_quant_conv = torch_neuronx.dynamic_batch(pipe.vae.post_quant_conv)

    LOG.info("Model loaded")

    return pipe


@timeutils.timeit
def load_gpu_diffusion(model):
    LOG.info("Loading gpu model %s", model)
    pipe = StableDiffusionPipeline.from_pretrained(model)
    if not torch.cuda.is_available():
        raise Exception('Unable to load model to gpu, no device available')
    pipe.to("cuda")
    return pipe


@timeutils.timeit
def load_prompts(count):
    LOG.info("Loading diffusion prompts dataset")
    prompts = datasets.load_dataset("Gustavosta/Stable-Diffusion-Prompts", split='train')
    LOG.info("Selecting random prompts")
    select_indices = np.random.choice(range(0, len(prompts)), count, replace=False)
    LOG.debug("%d samples, selected indices %s", len(select_indices), select_indices)
    return prompts.select(select_indices)


def decode_latents(self, latents):
    latents = latents.to(torch.float)
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
