# pip install controlnet-aux

# script to run the SoftEdge ControlNet

import torch
from PIL import Image
from controlnet_aux import HEDdetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# load and resize the baseline image
image_path = "/mnt/gcs_dataset/in.jpg"
original_image = load_image(image_path).resize((512, 512)) 

# extract Soft Edges using HED (Holistically-Nested Edge Detection)
processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
softedge_image = processor(original_image)

# save the SoftEdge map to verify the organic gradients
softedge_image.save("/mnt/gcs_dataset/out_debug_softedge.png") 

# initialize the SoftEdge ControlNet and Stable Diffusion Pipeline
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge", 
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)

# optimization for the T4 GPU
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# prompting strategy 
# keeps the precise morphological descriptions 
prompt = "a highly detailed, medical realism close up photo of two fingers, dark skin tone, distinct scaly skin lesion on the lower finger, natural fingernails, clear lighting, clinical photography"
negative_prompt = "jewelry, rings, metal bands, silver, gold, shiny, blurry, distorted, unnatural colors, low resolution, extra fingers, deformed anatomy"

# generate the augmented image
output = pipe(
    prompt, 
    image=softedge_image,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    controlnet_conditioning_scale=0.85 
).images[0]

output.save("/mnt/gcs_dataset/out_final_softedge.png")