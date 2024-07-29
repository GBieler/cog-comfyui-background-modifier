# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        background_prompt = workflow["13"]["inputs"]
        background_prompt["text"] = kwargs["background_prompt"]

        light_prompt = workflow["15"]["inputs"]
        light_prompt["text"] = kwargs['light_prompt']

        denoise_strength = workflow["46"]["inputs"]
        denoise_strength["value"] = kwargs["denoise_strength"]

        IC_light_scheduler_seed = workflow["91"]["inputs"]
        IC_light_scheduler_seed["seed"] = kwargs["seed_IC_light"]

        IP_adapter_scheduler_seed = workflow["45"]["inputs"]
        IP_adapter_scheduler_seed["seed"] = kwargs["seed_IP_adapter"]

        if kwargs['subject_image_filename']:
            load_subject_image = workflow["1"]["inputs"]
            load_subject_image["image"] = kwargs["subject_image_filename"]

        if kwargs['background_image_filename']:
            load_background_image = workflow["8"]["inputs"]
            load_background_image["image"] = kwargs["background_image_filename"]
        
        if kwargs['light_mask_filename']:
            load_light_mask_image = workflow["9"]["inputs"]
            load_light_mask_image["image"] = kwargs["light_mask_filename"]

    def predict(
        self,
        subject_image: Path = Input(
            description="Image with the subject",
            default=None,
        ),
        background_image: Path = Input(
            description="Image to use as a reference for the background (optional)",
            default=None,
        ),
        background_prompt: str = Input(
            description="Description of the background (optional)",
            default="",
        ),
        light_prompt: str = Input(
            description="Description of the lighting (optional)",
            default="",
        ),
        light_mask: Path = Input(
            description="Mask to use as a reference for the lighting (optional)",
            default="https://projectsgb.s3.amazonaws.com/test_images/lightMask.png",
        ),
        denoise_strength: float = Input(
            description="Background variation strength",
            default=0.7,
            ge=0.1,
            le=1,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        seed_IC_light = None
        seed_IP_adapter = None
        seed_IC_light = seed_helper.generate(seed_IC_light)
        seed_IP_adapter  = seed_helper.generate(seed_IP_adapter)

        
        subject_image_filename = None
        background_image_filename = None
        light_mask_filename = None
        if subject_image:
            subject_image_filename = self.filename_with_extension(subject_image, "image")
            self.handle_input_file(subject_image, subject_image_filename)
        else:
            raise ValueError("Please upload a subject image before continuing")
        if background_image:
            background_image_filename = self.filename_with_extension(background_image, "image")
            self.handle_input_file(background_image, background_image_filename)
        if light_mask:
            light_mask_filename = self.filename_with_extension(light_mask, "image")
            self.handle_input_file(light_mask, light_mask_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            background_prompt=background_prompt,
            light_prompt=light_prompt,
            subject_image_filename=subject_image_filename,
            background_image_filename=background_image_filename,
            light_mask_filename=light_mask_filename,
            denoise_strength=denoise_strength,
            seed_IC_light=seed_IC_light,
            seed_IP_adapter=seed_IP_adapter
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        output_format="png"
        output_quality=80
        

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
