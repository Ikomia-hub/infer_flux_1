import torch
from huggingface_hub import login
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize


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
        token = None
        repo = "black-forest-labs/FLUX.1-dev"

    if parameters.model_name=='flux1-schnell-fp8':
        model_lk = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8.safetensors"
        token = parameters.token
        repo = "black-forest-labs/FLUX.1-schnell"

    return model_lk, repo, token


def load_pipe(param, folder_path):
    model_link, bfl_repo, token_hf = get_model_info(param)

    float16_support, bfloat16_support = check_float16_and_bfloat16_support()
    dtype = torch.bfloat16 if bfloat16_support else torch.float16 \
                        if float16_support else torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('Preparing models for FP8 inference, this may take a while...')
    transformer = FluxTransformer2DModel.from_single_file(
                                            model_link,
                                            torch_dtype=dtype,
                                            cache_dir=folder_path)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    text_encoder_2 = T5EncoderModel.from_pretrained(
                                        bfl_repo,
                                        subfolder="text_encoder_2",
                                        torch_dtype=dtype,
                                        cache_dir=folder_path,
                                        token=token_hf)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    pipe = FluxPipeline.from_pretrained(
                                bfl_repo,
                                transformer=None,
                                text_encoder_2=None,
                                torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    pipe.to(device)
    if param.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    if param.vae_enable_slicing:
        pipe.vae.enable_slicing()

    if param.vae_enable_tiling:
        pipe.vae.enable_tiling()

    return pipe