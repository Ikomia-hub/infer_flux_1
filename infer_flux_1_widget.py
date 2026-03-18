import os
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_flux_1.infer_flux_1_process import InferFlux1Param
from infer_flux_1.utils.widget_utils import Autocomplete
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferFlux1Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferFlux1Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Model name
        model_list_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'utils', "model_list.txt")
        model_list_file = open(model_list_path, "r")

        model_list = model_list_file.read()
        model_list = model_list.split("\n")
        self.combo_model = Autocomplete(
            model_list, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.grid_layout.addWidget(self.combo_model, 0, 1)
        self.grid_layout.addWidget(self.label_model, 0, 0)
        self.combo_model.setCurrentText(self.parameters.model_name)
        model_list_file.close()

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(
            self.grid_layout, "Prompt", self.parameters.prompt)

        # Number of inference steps
        self.spin_number_of_steps = pyqtutils.append_spin(
            self.grid_layout,
            "Number of steps",
            self.parameters.num_inference_steps,
            min=1, step=1
        )
        # Guidance scale
        self.spin_guidance_scale = pyqtutils.append_double_spin(
            self.grid_layout,
            "Guidance scale",
            self.parameters.guidance_scale,
            min=0, step=0.1, decimals=1
        )
        # Image output size
        self.spin_height = pyqtutils.append_spin(
            self.grid_layout,
            "Image height",
            self.parameters.height,
            min=128, step=1
        )

        # Number of inference steps
        self.spin_width = pyqtutils.append_spin(
            self.grid_layout,
            "Image width",
            self.parameters.width,
            min=128, step=1
        )

        # Seed
        self.spin_seed = pyqtutils.append_spin(
            self.grid_layout,
            "Seed",
            self.parameters.seed,
            min=-1, step=1
        )

        # Number output images
        self.spin_num_images_per_prompt = pyqtutils.append_spin(
            self.grid_layout,
            "Number of output(s)",
            self.parameters.num_images_per_prompt,
            min=1, max=6)

        # Cuda
        self.check_enable_model_cpu_offload = pyqtutils.append_check(self.grid_layout,
                                                                     "Enable model CPU offload",
                                                                     self.parameters.enable_model_cpu_offload)

        # Cuda
        self.check_vae_enable_slicing = pyqtutils.append_check(self.grid_layout,
                                                               "VAE enable slicing",
                                                               self.parameters.vae_enable_slicing)

        # Cuda
        self.check_vae_enable_tiling = pyqtutils.append_check(self.grid_layout,
                                                              "VAE enable tiling",
                                                              self.parameters.vae_enable_tiling)
        # token
        self.edit_token = pyqtutils.append_edit(
            self.grid_layout, "Token Hugging Face", self.parameters.token)

        self.label_hyp = QLabel("LoRA weights")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.lora_weight_file,
            tooltip="Select file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.emit_apply(self.parameters)
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.token = self.edit_token.text()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.num_inference_steps = self.spin_number_of_steps.value()
        self.parameters.guidance_scale = self.spin_guidance_scale.value()
        self.parameters.width = self.spin_width.value()
        self.parameters.height = self.spin_height.value()
        self.parameters.seed = self.spin_seed.value()
        self.parameters.enable_model_cpu_offload = self.check_enable_model_cpu_offload.isChecked()
        self.parameters.vae_enable_slicing = self.check_vae_enable_slicing.isChecked()
        self.parameters.vae_enable_tiling = self.check_vae_enable_tiling.isChecked()
        self.parameters.lora_weight_file = self.browse_weight_file.path


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferFlux1WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_flux_1"

    def create(self, param):
        # Create widget object
        return InferFlux1Widget(param, None)
