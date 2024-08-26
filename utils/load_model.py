import torch
from huggingface_hub import login
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize, QuantizedDiffusersModel, QuantizedTransformersModel
import os


class QuantizedFlux2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

class QuantizedT5Model(QuantizedTransformersModel):
    auto_class = T5EncoderModel


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
    if parameters.model_name=='flux1-dev-fp8':
        model_lk = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
        if parameters.token:
            login(token=parameters.token)
        else:
            print('Please use a Hugging Face token to use the FLUX dev model')
        repo = "black-forest-labs/FLUX.1-dev"
        model_version = "dev"

    if parameters.model_name=='flux1-schnell-fp8':
        model_lk = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8.safetensors"
        repo = "black-forest-labs/FLUX.1-schnell"
        model_version = "schnell"

    return model_lk, repo, model_version

def load_pipe(param, folder_path):
    model_link, bfl_repo, model_type = get_model_info(param)

    float16_support, bfloat16_support = check_float16_and_bfloat16_support()
    dtype = torch.bfloat16 if bfloat16_support else torch.float16 \
                        if float16_support else torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check if the folder exists and that there is a safetensors model in the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    safetensor_model_path = os.path.join(folder_path, f"flux_{model_type}_qfp8", "diffusion_pytorch_model-00001-of-00002.safetensors")
    if not os.path.exists(safetensor_model_path):
        print(f"Preparing the FLUX {model_type} model for FP8 inference, this may take a while...")
        transformer = FluxTransformer2DModel.from_single_file(
            model_link,
            torch_dtype=dtype,
            cache_dir=folder_path
            )
        qtransformer = QuantizedFlux2DModel.quantize(transformer, weights=qfloat8)

        print(f"Saving quantized FLUX {model_type} model...")
        qtransformer.save_pretrained(f"{folder_path}/flux_{model_type}_qfp8")
    else:
        print(f'Quantized FLUX {model_type} available at {folder_path}/flux_{model_type}_qfp8')

    print('Preparing the T5 encoder  model for FP8 inference, this may take a while...')
    text_encoder_2 = T5EncoderModel.from_pretrained(
                                        bfl_repo,
                                        subfolder="text_encoder_2",
                                        torch_dtype=dtype,
                                        cache_dir=folder_path)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    # Loading
    print("Loading FLUX pipeline...")
    qtransformer = QuantizedFlux2DModel.from_pretrained(f"{folder_path}/flux_{model_type}_qfp8")
    qtransformer.to(device=device, dtype=dtype)

    pipe = FluxPipeline.from_pretrained(
                                bfl_repo,
                                transformer=None,
                                text_encoder_2=None,
                                torch_dtype=dtype)
    pipe.transformer = qtransformer
    pipe.text_encoder_2 = text_encoder_2
    pipe.to(device)

    if param.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    if param.vae_enable_slicing:
        pipe.vae.enable_slicing()

    if param.vae_enable_tiling:
        pipe.vae.enable_tiling()

    return pipe