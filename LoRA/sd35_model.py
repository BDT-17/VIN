"""SD3.5 CityPersons augmentation: model and pipeline loading."""

import csv
import gc
import json
from datetime import datetime
import math
import numpy as np
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops

try:
    import cv2
except ImportError:
    cv2 = None

from sd35_config import *
from sd35_utils import clear_cuda

os.environ["DIFFUSERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message="Flax classes are deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.*")

from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()
try:
    from diffusers import StableDiffusion3InpaintPipeline
except ImportError:
    StableDiffusion3InpaintPipeline = None

PIPELINE_LOAD_LOCK = Lock()


def lora_metadata():
    return {
        "lora_enabled": bool(LORA_ENABLED),
        "lora_path": "" if LORA_PATH is None else str(LORA_PATH),
        "lora_weight_name": "" if LORA_WEIGHT_NAME is None else str(LORA_WEIGHT_NAME),
        "lora_adapter_name": str(LORA_ADAPTER_NAME),
        "lora_scale": float(LORA_SCALE),
        "lora_fuse": bool(LORA_FUSE),
    }


def apply_lora_adapter(pipe):
    if not LORA_ENABLED:
        return pipe
    if not LORA_PATH:
        raise ValueError("LORA_ENABLED=True but LORA_PATH is not set.")
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(
            "This diffusers pipeline does not expose load_lora_weights(). "
            "Use a diffusers version with SD3 LoRA loader support."
        )
    load_kwargs = {"adapter_name": LORA_ADAPTER_NAME}
    if LORA_WEIGHT_NAME:
        load_kwargs["weight_name"] = LORA_WEIGHT_NAME
    pipe.load_lora_weights(str(LORA_PATH), **load_kwargs)
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[float(LORA_SCALE)])
    elif float(LORA_SCALE) != 1.0:
        raise RuntimeError(
            "LORA_SCALE requires set_adapters() support in the active diffusers pipeline."
        )
    if LORA_FUSE:
        if not hasattr(pipe, "fuse_lora"):
            raise RuntimeError("LORA_FUSE=True but the pipeline does not expose fuse_lora().")
        pipe.fuse_lora(adapter_names=[LORA_ADAPTER_NAME], lora_scale=float(LORA_SCALE))
    print(
        f"Loaded LoRA adapter {LORA_ADAPTER_NAME!r} from {LORA_PATH} "
        f"with scale={float(LORA_SCALE):.3f}"
    )
    return pipe

def resolve_augmentation_devices():
    if AUGMENTATION_DEVICES:
        return AUGMENTATION_DEVICES
    if not torch.cuda.is_available():
        return ["cpu"]
    if USE_ALL_GPUS_FOR_AUGMENTATION:
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return [TRAIN_DEVICE]


def build_img2img_pipeline(backend=MODEL_BACKEND, device=TRAIN_DEVICE):
    if backend == "sd35":
        pipeline_cls = StableDiffusion3Img2ImgPipeline
        model_id = SD35_MODEL_ID
        kwargs = {"torch_dtype": torch.float16, "use_safetensors": True, "low_cpu_mem_usage": True}
        if not USE_T5:
            kwargs.update({"text_encoder_3": None, "tokenizer_3": None})
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)

    with PIPELINE_LOAD_LOCK:
        try:
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs.pop("text_encoder_3", None)
            kwargs.pop("tokenizer_3", None)
            kwargs.pop("low_cpu_mem_usage", None)
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
        pipe = apply_lora_adapter(pipe)

        if str(device).startswith("cuda") and USE_MODEL_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
            gpu_id = torch.device(device).index or 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            print(f"Enabled model CPU offload for {device}")
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    return pipe


def build_inpaint_pipeline(backend=MODEL_BACKEND, device=TRAIN_DEVICE):
    if backend == "sd35":
        if StableDiffusion3InpaintPipeline is None:
            raise ImportError("StableDiffusion3InpaintPipeline is not available in this diffusers version.")
        pipeline_cls = StableDiffusion3InpaintPipeline
        model_id = SD35_MODEL_ID
        kwargs = {"torch_dtype": torch.float16, "use_safetensors": True, "low_cpu_mem_usage": True}
        if not USE_T5:
            kwargs.update({"text_encoder_3": None, "tokenizer_3": None})
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)

    with PIPELINE_LOAD_LOCK:
        try:
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs.pop("text_encoder_3", None)
            kwargs.pop("tokenizer_3", None)
            kwargs.pop("low_cpu_mem_usage", None)
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
        pipe = apply_lora_adapter(pipe)

        if str(device).startswith("cuda") and USE_MODEL_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
            gpu_id = torch.device(device).index or 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            print(f"Enabled model CPU offload for {device}")
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    return pipe


