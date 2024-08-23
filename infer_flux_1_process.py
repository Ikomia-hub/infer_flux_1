import copy
import torch
import random
import os
import numpy as np
from ikomia import core, dataprocess, utils
from infer_flux_1.utils.load_model import load_pipe
# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFlux1Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "flux1-schnell-fp8"
        self.prompt = "A cat holding a sign that says hello world, outdoor, garden"
        self.token = ""
        self.cuda = torch.cuda.is_available()
        self.guidance_scale = 0
        self.num_inference_steps = 4
        self.seed = -1
        self.width = 1024
        self.height = 1024
        self.num_images_per_prompt = 1
        self.enable_model_cpu_offload = True
        self.vae_enable_slicing = False
        self.vae_enable_tiling = False
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = param_map["prompt"]
        self.token = param_map["token"]
        self.cuda = utils.strtobool(param_map["cuda"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.seed = int(param_map["seed"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.width = int(param_map["width"])
        self.height = int(param_map["height"])
        self.num_images_per_prompt = int(param_map["num_images_per_prompt"])
        self.enable_model_cpu_offload = utils.strtobool(param_map["enable_model_cpu_offload"])
        self.vae_enable_slicing = utils.strtobool(param_map["vae_enable_slicing"])
        self.vae_enable_tiling = utils.strtobool(param_map["vae_enable_tiling"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["token"] = str(self.token)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["seed"] = str(self.seed)
        param_map["width"] = str(self.width)
        param_map["height"] = str(self.height)
        param_map["num_images_per_prompt"] = str(self.num_images_per_prompt)
        param_map["enable_model_cpu_offload"] = str(self.enable_model_cpu_offload)
        param_map["vae_enable_slicing"] = str(self.vae_enable_slicing)
        param_map["vae_enable_tiling"] = str(self.vae_enable_tiling)
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFlux1(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferFlux1Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None
        self.generator = None
        self.seed = None
        self.width = 1024
        self.height = 1024
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1
    
    def set_generator(self, seed_num):
            if seed_num == -1:
                self.seed = random.randint(0, 191965535)
            else:
                self.seed = seed_num
            self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if self.pipe is None:
            self.pipe = load_pipe(param, self.model_folder)

        
        self.set_generator(param.seed)

        # Inference
        with torch.no_grad():
            results = self.pipe(
                            param.prompt,
                            guidance_scale = param.guidance_scale,
                            generator = self.generator,
                            num_inference_steps = param.num_inference_steps,
                            width=self.width,
                            height=self.height
                            ).images

        print(f"Prompt:\t{param.prompt}\nSeed:\t{self.seed}")

        # Set image output
        if len(results) > 1:
            for i, image in enumerate(results):
                self.add_output(dataprocess.CImageIO())
                img = np.array(image)
                output = self.get_output(i)
                output.set_image(img)
        else:
            image = np.array(results[0])
            output_img = self.get_output(0)
            output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()



# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFlux1Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_flux_1"
        self.info.short_description = "Flux is a series of text-to-image generation models based on diffusion transformers"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Andreas Blattmann, Axel Sauer, Dominik Lorenz, Dustin Podell, " \
                            "Frederic Boesel, Harry Saini, Jonas Müller, Kyle Lacey, " \
                            "Patrick Esser, Robin Rombach, Sumith Kulal, Tim Dockhorn, " \
                            "Yam Levi, Zion English"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "Apache License 2.0"

        # Ikomia API compatibility
        # self.info.min_ikomia_version = "0.11.1"
        # self.info.max_ikomia_version = "0.11.1"

        # Python compatibility
        self.info.min_python_version = "3.10.0"
        # self.info.max_python_version = "3.11.0"
        # URL of documentation
        self.info.documentation_link = "https://blackforestlabs.ai/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_flux_1"
        self.info.original_repository = "https://github.com/black-forest-labs/flux"

        self.info.keywords = "Diffusion, Hugging Face, Black Forest,text-to-image, Generative"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"

    def create(self, param=None):
        # Create algorithm object
        return InferFlux1(self.info.name, param)
