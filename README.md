# CityPersons Pedestrian Augmentation With SD3.5

Research baseline for CityPersons pedestrian augmentation using **Stable Diffusion 3.5 Medium** and **YOLOv8m-seg**.

Current pipeline: **V5 context-person-composite with img2img**.

```text
Dataset scanner
-> Context crop img2img generation
-> YOLO person segmentation
-> Scale correction
-> Edge / color / brightness / shadow blending
-> Person-only alpha composite
-> YOLO validation + retry
-> Image outputs + manifest
```

## Main Files

- `sd35_run.ipynb`: short Kaggle runner. It clones/pulls this GitHub repo and imports the modules from the cloned repo.
- `sd3.5-agumentation-scale-correction-clean.ipynb`: self-contained Kaggle notebook. It writes the Python modules to `/kaggle/working` with `%%writefile`, then imports and runs them.
- `sd35_config.py`: configuration.
- `sd35_data.py`: dataset scan and preview.
- `sd35_utils.py`: preprocessing, placement, masks, scale/depth helpers.
- `sd35_model.py`: SD3.5 pipeline loading.
- `sd35_evaluation.py`: YOLO-seg, validation, retry policy.
- `sd35_pipeline.py`: generation, paste, edge correction, compositing.
- `sd35_runner.py`: job building, augmentation runner, manifest, autotune, export.

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

You do **not** need to upload or copy `sd35_*.py` manually when using this workflow.

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

The dataset is not stored in this repo. Add the CityPersons Kaggle dataset as notebook input. The config searches common Kaggle paths under `/kaggle/input`.

## Important Configuration

Edit `sd35_config.py` for clone-based runs, or edit the `%%writefile /kaggle/working/sd35_config.py` cell in the self-contained notebook.

```python
RUN_PRESET = "batch"  # smoke | quality | batch
PARAMETER_OVERRIDES = {}
USE_ALL_GPUS_FOR_AUGMENTATION = True
CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"
```

Smoke run defaults:

```python
SMOKE_IMAGES = 10
SMOKE_SPLITS = ["train"]
```

Main defaults:

- `BACKGROUND_PRESERVATION_MODE = "context_person_composite"`
- `CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"`
- `RESOLUTION = 512`
- `USE_T5 = False`
- `USE_SEAMLESS_CLONE = False`
- `CONTEXT_PERSON_MASK_THRESHOLD = 0.40`
- `PERSON_MASK_TRIM_FRINGE_PIXELS = 1`

## Edge Handling

The pipeline applies edge correction **after** the person has been pasted:

```text
paste person
-> crop pasted result
-> horizontal-row background mean around person
-> blur/mean edge correction
-> paste corrected crop back
```

Current edge correction uses **horizontal mean only**. It does not blend local mean.

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

This is a research baseline focused on scale correction, grounding, occlusion ordering, blending, validation, and autotune. It is not a production augmentation service.
