<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_flux_1</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_flux_1">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_flux_1">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_flux_1/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_flux_1.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Flux is a series of text-to-image generation models utilizing diffusion transformers, developed by Black Forest Labs, the ex-team members of Stable Diffusion.

![illustration](https://github.com/black-forest-labs/flux/blob/main/assets/grid.jpg?raw=true)

*This FLUX1 algorithm runs FP8 inference and requires about 16 GB of VRAM and 30GB of CPU memory.*

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_flux_1", auto_connect=False)

# Run directly on your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio
Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'flux1-dev': Name of the stable diffusion model. Other model available:
    - flux1-schnell
- **prompt** (str) - default 'A cat holding a sign that says hello world, outdoor, garden' : Text prompt to guide the image generation.
- **num_inference_steps** (int) - default '4': Number of inference steps
- **guidance_scale** (float) - default '0.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **height** (int) - default '1024': The height in pixels of the generated image.
- **width** (int) - default '1024': The width in pixels of the generated image.
- **num_images_per_prompt** (int) - default '1': Number of generated image(s).
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.
- **token** (str) - default '' : Your Hugging Face user token ('Read' rights). 
- **enable_model_cpu_offload** (bool) - default 'False' : Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. This method moves one whole model at a time to the GPU when its forward method is called, and the model remains in GPU until the next model runs. 
- **vae_enable_slicing** (bool) - default 'False' : Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
- **vae_enable_tiling** (bool) - default 'False' : Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in several steps. This is useful to save a large amount of memory and to allow the processing of larger images.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_flux_1", auto_connect=False)

algo.set_parameters({
    'model_name': 'flux1-schnell',
    'prompt': 'A cat holding a sign that says hello world',
    'num_inference_steps': '4',
    'guidance_scale': '0',
    'seed': '-1',
    'width': '1024',
    'height': '1024',
    'num_images_per_prompt':'1',
    'token': '[YOUR HF USER TOKEN]', # Only for the dev model version
    'enable_model_cpu_offload': 'False'
    })

# Generate your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

### Recommended setting
- **FLUX1 dev** : num_inference_steps between 20 and 50, guidance_scale: 3.5
- **FLUX1 schnell** : num_inference_steps between 4, guidance_scale: 0

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_flux_1", auto_connect=False)

# Run  
wf.run()

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```


## :fast_forward: Advanced usage 

To run Stable Diffusion 3 you need to:
1. Generate your Hugging Face [access token](https://huggingface.co/docs/hub/security-tokens) (Type: Read)
2. Share your contact info to Hugging Face in order to access the [Flux dev model](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)