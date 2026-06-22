# Background-Preserving Object Insertion Augmentation

Research baseline for building a data augmentation pipeline that **adds new objects to existing images while preserving the original background**.

The current implementation uses **Stable Diffusion 3.5 Medium** for candidate generation and **YOLOv8m-seg** for object extraction/validation. CityPersons pedestrian insertion is the reference experiment in this repository, but the target task is broader than CityPersons:

```text
given an image dataset
-> choose a target object class and insertion policy
-> generate candidate object(s) in local scene context
-> segment only the newly generated object pixels
-> correct scale, placement, and occlusion
-> paste the object back onto the original image
-> validate, retry, and log metadata
```

The central design rule is simple: **the source image is the trusted background; generated background pixels are not trusted**.

## Current Reference Pipeline

The active implementation is **V5 context-object-composite**. In the current code and notebooks this is still named `context_person_composite` because the reference class is pedestrian.

```text
Dataset scanner
-> Object placement proposal
-> Context img2img generation
-> Target-object segmentation
-> Perspective / scale correction
-> Edge / color / brightness / shadow blending
-> Object-only alpha composite
-> Detector validation + retry
-> Image outputs + manifest
```

## Main Files

- `sd35_run.ipynb`: short Kaggle runner. It clones/pulls this GitHub repo and imports the modules from the cloned repo.
- `sd3.5-agumentation-scale-correction-clean.ipynb`: self-contained Kaggle notebook. It writes the Python modules to `/kaggle/working` with `%%writefile`, then imports and runs them.
- `sd35_config.py`: configuration, presets, generation parameters, placement thresholds, validation thresholds.
- `sd35_data.py`: dataset scan and preview.
- `sd35_utils.py`: preprocessing, placement, masks, scale/depth helpers.
- `sd35_model.py`: SD3.5 pipeline loading.
- `sd35_evaluation.py`: detector/segmenter validation and retry policy.
- `sd35_pipeline.py`: generation, object paste, edge correction, compositing.
- `sd35_edge_harmonization.py`: boundary-only edge harmonization.
- `sd35_runner.py`: job building, augmentation runner, manifest, autotune, export.

## Scope

This repo is not only about CityPersons. CityPersons is the first concrete benchmark because pedestrians expose many hard cases:

- strong perspective scale changes;
- foot grounding;
- foreground occlusion;
- small and distant instances;
- downstream detector sensitivity;
- strict background preservation requirements.

To adapt the pipeline to another dataset or object class, the key parts to swap are:

- dataset scanner and label loader;
- object prompts and variant definitions;
- detector/segmenter for the target object class;
- scale policy for the target object geometry;
- placement constraints for valid object locations;
- validation criteria and manifest fields.

## Run On Kaggle

### Recommended: Clone-Based Runner

Open a new Kaggle notebook with Internet ON and GPU enabled, then use `sd35_run.ipynb`.

The runner does this automatically:

```python
REPO_URL = "https://github.com/BDT-17/VIN.git"
REPO_DIR = Path("/kaggle/working/VIN")

if REPO_DIR.exists():
    %cd /kaggle/working/VIN
    !git pull
else:
    !git clone {REPO_URL} {REPO_DIR}
    %cd /kaggle/working/VIN

PROJECT_DIR = REPO_DIR / "notebooks"
if not (PROJECT_DIR / "sd35_config.py").exists():
    PROJECT_DIR = REPO_DIR

%cd {PROJECT_DIR}
```

Then it imports the modules from `PROJECT_DIR`:

```python
from sd35_config import *
from sd35_data import *
from sd35_utils import *
from sd35_model import *
from sd35_evaluation import *
from sd35_pipeline import *
from sd35_runner import *
```

Run the notebook cells in order:

```text
1. Install Dependencies
2. Clone Or Update Repo
3. Imports
4. Runtime Check
5. Hugging Face Login
6. Dataset Scan
7. Smoke Run
8. Export Outputs, optional
```

For Hugging Face access, add a Kaggle secret named `HF_TOKEN` before running the login cell:

```text
Kaggle Notebook -> Add-ons -> Secrets -> Add secret
Name: HF_TOKEN
Value: your Hugging Face access token
```

The notebooks read this secret with `UserSecretsClient().get_secret("HF_TOKEN")` and call `login(token=hf_token)`. For local runs, you can alternatively set an environment variable named `HF_TOKEN`.

### Alternative: Self-Contained Notebook

Use `sd3.5-agumentation-scale-correction-clean.ipynb` if you want to upload a single notebook only. This notebook embeds the module source in `%%writefile` cells and writes modules into `/kaggle/working` before importing them.

## Dataset

The dataset is not stored in this repo.

The current config searches common CityPersons Kaggle paths under `/kaggle/input`, but this is a reference setup rather than a hard project boundary. For another dataset, update:

- `DATASET_ROOT_CANDIDATES`
- `DATASET_SPLIT_DIRS`
- label parsing helpers in `sd35_data.py` / `sd35_utils.py`
- target object class IDs and validation thresholds

## Important Configuration

Edit `sd35_config.py` for clone-based runs, or edit the `%%writefile /kaggle/working/sd35_config.py` cell in the self-contained notebook.

```python
RUN_PRESET = "batch"  # smoke | quality | batch
PARAMETER_OVERRIDES = {}
USE_ALL_GPUS_FOR_AUGMENTATION = True
BACKGROUND_PRESERVATION_MODE = "context_person_composite"
CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"
```

The naming is still pedestrian-specific in code, but the architecture is object-insertion oriented.

Main defaults:

- `RESOLUTION = 512`
- `USE_T5 = False`
- `USE_SEAMLESS_CLONE = False`
- `CONTEXT_PERSON_MASK_THRESHOLD = 0.40`
- `PERSON_MASK_TRIM_FRINGE_PIXELS = 1`
- `EDGE_HARMONIZATION_ENABLED = True`

## Edge Handling

The pipeline applies edge correction after the object has been pasted:

```text
paste segmented object
-> crop pasted result
-> estimate local background tone around object boundary
-> blur / color / brightness correction on the boundary band
-> paste corrected crop back
```

The goal is to reduce visible pasted-object artifacts without letting background pixels overwrite the inserted object core.

## GPU Memory

The runner processes devices sequentially to keep VRAM stable on Kaggle T4:

```text
load pipeline -> run shard -> del pipe -> clear_cuda() -> next device
```

## Outputs

The run can produce:

- augmented images;
- comparison pairs;
- optional debug images;
- manifest rows;
- rejection histogram;
- quality metrics;
- autotune snapshots;
- optional zip artifact from `export_outputs()`.

## Status

This is a research baseline for **background-preserving object insertion augmentation**. It is not a production augmentation service. The current implementation is strongest for pedestrians because that is the reference class, but the intended abstraction is a general object insertion pipeline.
