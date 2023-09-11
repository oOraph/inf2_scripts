from optimum.neuron import NeuronStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}

input_shapes = {"batch_size": 1, "height": 512, "width": 512}  

stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_id, export=True, **compiler_args, **input_shapes)

# Save locally or upload to the HuggingFace Hub

save_directory = "sd_neuron/"

stable_diffusion.save_pretrained(save_directory)
import pdb
pdb.set_trace()
stable_diffusion.push_to_hub(
    save_directory, repository_id="raphael-gl/my-neuron-repo", use_auth_token=True
)

