# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.handle_weights(
            {},
            weights_to_download=[
                "appearance_feature_extractor.safetensors",
                "landmark.onnx",
                "motion_extractor.safetensors",
                "spade_generator.safetensors",
                "stitching_retargeting_module.safetensors",
                "warping_module.safetensors",
            ],
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

    def update_workflow(self, workflow, **kwargs):
        load_video = workflow["8"]["inputs"]
        load_video["video"] = kwargs["driving_filename"]
        load_video["frame_load_cap"] = kwargs["frame_load_cap"]
        load_video["select_every_n_frames"] = kwargs["select_every_n_frames"]

        live_portrait = workflow["30"]["inputs"]
        live_portrait["dsize"] = kwargs["dsize"]
        live_portrait["scale"] = kwargs["scale"]
        live_portrait["vx_ratio"] = kwargs["vx_ratio"]
        live_portrait["vy_ratio"] = kwargs["vy_ratio"]
        live_portrait["lip_zero"] = kwargs["lip_zero"]
        live_portrait["eye_retargeting"] = kwargs["eye_retargeting"]
        live_portrait["eyes_retargeting_multiplier"] = kwargs[
            "eyes_retargeting_multiplier"
        ]
        live_portrait["lip_retargeting"] = kwargs["lip_retargeting"]
        live_portrait["lip_retargeting_multiplier"] = kwargs[
            "lip_retargeting_multiplier"
        ]
        live_portrait["stitching"] = kwargs["stitching"]
        live_portrait["relative"] = kwargs["relative"]

    def predict(
        self,
        face_image: Path = Input(
            description="An image with a face",
        ),
        driving_video: Path = Input(
            description="A video to drive the animation",
        ),
        video_frame_load_cap: int = Input(
            description="The maximum number of frames to load from the driving video. Set to 0 to use all frames.",
            default=64,
        ),
        video_select_every_n_frames: int = Input(
            description="Select every nth frame from the driving video. Set to 1 to use all frames.",
            default=1,
        ),
        live_portrait_dsize: int = Input(
            description="Size of the output image", default=512, ge=64, le=2048
        ),
        live_portrait_scale: float = Input(
            description="Scaling factor for the face", default=2.3, ge=1.0, le=4.0
        ),
        live_portrait_vx_ratio: float = Input(
            description="Horizontal shift ratio", default=0, ge=-1.0, le=1.0
        ),
        live_portrait_vy_ratio: float = Input(
            description="Vertical shift ratio", default=-0.125, ge=-1.0, le=1.0
        ),
        live_portrait_lip_zero: bool = Input(
            description="Enable lip zero", default=True
        ),
        live_portrait_eye_retargeting: bool = Input(
            description="Enable eye retargeting", default=False
        ),
        live_portrait_eyes_retargeting_multiplier: float = Input(
            description="Multiplier for eye retargeting", default=1.0, ge=0.01, le=10.0
        ),
        live_portrait_lip_retargeting: bool = Input(
            description="Enable lip retargeting", default=False
        ),
        live_portrait_lip_retargeting_multiplier: float = Input(
            description="Multiplier for lip retargeting", default=1.0, ge=0.01, le=10.0
        ),
        live_portrait_stitching: bool = Input(
            description="Enable stitching", default=True
        ),
        live_portrait_relative: bool = Input(
            description="Use relative positioning", default=True
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        face_filename = self.filename_with_extension(face_image, "face")
        self.handle_input_file(face_image, face_filename)

        driving_filename = self.filename_with_extension(driving_video, "driving")
        self.handle_input_file(driving_video, driving_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            face_filename=face_filename,
            driving_filename=driving_filename,
            frame_load_cap=video_frame_load_cap,
            select_every_n_frames=video_select_every_n_frames,
            dsize=live_portrait_dsize,
            scale=live_portrait_scale,
            vx_ratio=live_portrait_vx_ratio,
            vy_ratio=live_portrait_vy_ratio,
            lip_zero=live_portrait_lip_zero,
            eye_retargeting=live_portrait_eye_retargeting,
            eyes_retargeting_multiplier=live_portrait_eyes_retargeting_multiplier,
            lip_retargeting=live_portrait_lip_retargeting,
            lip_retargeting_multiplier=live_portrait_lip_retargeting_multiplier,
            stitching=live_portrait_stitching,
            relative=live_portrait_relative,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR)
