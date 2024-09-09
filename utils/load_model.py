import torch
from huggingface_hub import login
from diffusers import DiffusionPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel
from torchao.quantization import quantize_, int8_weight_only
import os


def check_float16_and_bfloat16_support():
    if torch.cuda.is_available():
        gpu = torch.device('cuda')
        compute_capability = torch.cuda.get_device_capability(gpu)
        float16_support = compute_capability[0] >= 6  # Compute capability 6.0 or higher
        bfloat16_support = compute_capability[0] >= 8  # Compute capability 8.0 or higher
        if bfloat16_support:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return float16_support, bfloat16_support
    else:
        return False, False

def get_model_info(parameters):
    if parameters.model_name=='flux1-dev':
        if parameters.token:
            login(token=parameters.token)
        else:
            print('Please use a Hugging Face token to use the FLUX dev model')
        repo = "black-forest-labs/FLUX.1-dev"
        model_version = "dev"

    if parameters.model_name=='flux1-schnell':
        repo = "black-forest-labs/FLUX.1-schnell"
        model_version = "schnell"

    return repo, model_version

def load_pipe(param, folder_path):
    ckpt_id, model_type = get_model_info(param)

    float16_support, bfloat16_support = check_float16_and_bfloat16_support()
    dtype = torch.bfloat16 if bfloat16_support else torch.float16 \
                        if float16_support else torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check if the folder exists and that there is a safetensors model in the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Quantization process:
    ############ Diffusion Transformer ############
    transformer_pt = f"{folder_path}/flux_{model_type}_int8.pt"
    if not os.path.exists(transformer_pt):
        print(f"Preparing the FLUX {model_type} model for FP8 inference, this may take a while...")
        print("Quantization - step 1/4: FLUX transformer quantization")
        transformer = FluxTransformer2DModel.from_pretrained(
            ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=folder_path
        )
        quantize_(transformer, int8_weight_only())
        torch.save(transformer.state_dict(), transformer_pt)

    ############ Text Encoder ############
    te1_pt = f"{folder_path}/flux_{model_type}_te1_int8.pt"
    if not os.path.exists(te1_pt):
        print("Quantization - step 2/4: CLIP ViT large encoder quantization")
        text_encoder = CLIPTextModel.from_pretrained(
            ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=folder_path
        )
        quantize_(text_encoder, int8_weight_only())
        torch.save(text_encoder.state_dict(), te1_pt)

    ############ Text Encoder 2 ############
    te2_pt = f"{folder_path}/flux_{model_type}_te2_int8.pt"
    if not os.path.exists(te2_pt):
        print("Quantization - step 3/4: T5xxl encoder quantization")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, cache_dir=folder_path
        )
        quantize_(text_encoder_2, int8_weight_only())
        torch.save(text_encoder_2.state_dict(), te2_pt)

    ############ VAE ############
    vae_pt = f"{folder_path}/flux_{model_type}_vae_int8.pt"
    if not os.path.exists(vae_pt):
        print("Quantization - step 4/4: VAE quantization")
        vae = AutoencoderKL.from_pretrained(
            ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16, cache_dir=folder_path
        )
        quantize_(vae, int8_weight_only())
        torch.save(vae.state_dict(), vae_pt)

        torch.cuda.empty_cache()

    # Loading quantized models
    print("Loading FP8 quantized models")
    with torch.device("meta"):
        config = FluxTransformer2DModel.load_config(ckpt_id, subfolder="transformer")
        transformer = FluxTransformer2DModel.from_config(config).to(dtype)

    ############ Diffusion Transformer ############
    state_dict = torch.load(transformer_pt, map_location="cpu")
    transformer.load_state_dict(state_dict, assign=True)

    ############ Text Encoder ############
    text_encoder = CLIPTextModel.from_pretrained(
        ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    state_dict = torch.load(te1_pt, map_location="cpu")
    text_encoder.load_state_dict(state_dict, assign=True)

    ############ Text Encoder 2 ############
    text_encoder_2 = T5EncoderModel.from_pretrained(
        ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    state_dict = torch.load(te2_pt, map_location="cpu")
    text_encoder_2.load_state_dict(state_dict, assign=True)

    ############ VAE ############
    vae = AutoencoderKL.from_pretrained(
        ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    state_dict = torch.load(vae_pt, map_location="cpu")
    vae.load_state_dict(state_dict, assign=True)

    # Load pipeline
    print("Loading FLUX pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        ckpt_id, 
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    ).to(device)

    if param.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    if param.vae_enable_slicing:
        pipe.vae.enable_slicing()

    if param.vae_enable_tiling:
        pipe.vae.enable_tiling()

    return pipe