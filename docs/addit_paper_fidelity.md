# Add-it fidelity map — paper → code

This document maps every component of **Add-it** (Tewel et al., *Add-it:
Training-Free Object Insertion in Images with Pretrained Diffusion Models*,
ICLR 2025, arXiv:2411.07232) to its implementation in `addit(experimental)/`,
and records every place the SD3.5 port deviates from the FLUX.1-dev paper.

The re-implementation goal was **100% of the paper's method**. Where the paper
is backbone-specific (FLUX integer timesteps, FLUX layer indices, FLUX distilled
guidance), the *mechanism* is preserved and only the *constant* is re-derived
for SD3.5 Medium. Those are flagged `SD3.5-ADAPT` below and at the call site.

> **Two modes (`ADDIT_MODE`).** The attention-injection method documented in
> this file is the **`"faithful"`** mode. In practice on SD3.5 Medium it
> re-noises and decodes the whole image (background softens / blurs) and the
> subject mask often collapses to empty so nothing is inserted. The **default
> is now `"composite"`**: keep Add-it's affordance thesis (no input bbox — the
> model decides *where*) but run img2img → YOLOv8-seg the *added* person →
> paste only those pixels onto the byte-exact source. Background is preserved
> 100% and stays sharp because it never passes through the VAE/diffusion. The
> composite path is **not** the paper method — it trades the training-free
> attention mechanism for the repo's "source is the trusted background" rule.
> This document describes the faithful mode; see
> `AddItPipeline._add_object_composite` for the composite path.

## Files

| File | Paper section | Contents |
|------|---------------|----------|
| `addit_core.py` | §3.1–3.4 | extended-attention processor, γ root-solver, structure transfer / re-noising, Otsu + point sampling + latent blend |
| `addit_sam.py` | §3.4 / App A.1 | SAM-2 mask refinement (soft dependency, Otsu fallback) |
| `addit_config.py` | §3 / App A.1 | all constants, expressed as SD3.5 fractions |
| `addit_pipeline.py` | §3.2–3.5 | the full denoise loop, subject-token localisation, step-by-step |
| `addit_run.ipynb` | — | Kaggle runner (free placement, no bbox) |

## Component-by-component

### §3.1 MM-DiT joint attention
- Paper: DiT blocks run joint self-attention over `[text, image]` tokens.
- Code: `AddItJointAttnProcessor` wraps SD3.5's `JointAttnProcessor2_0`, keeping
  the stock `[text | image]` concatenation and `to_out` / `to_add_out` split.
- **SD3.5-ADAPT**: FLUX has *multi-stream* and *single-stream* block types; SD3.5
  has a single `JointTransformerBlock` type (24 of them). The single/multi
  distinction therefore collapses to one processor class.

### §3.2 Weighted extended self-attention (eq. 3)
- Paper eq. 3: `A = softmax([Q_p,Q_t]·[γ_s·K_src, γ_p·K_p, γ_t·K_t]ᵀ/√d)`,
  `h = A·[V_src,V_p,V_t]`. The **keys** are weighted; values are not.
- Code: `AddItJointAttnProcessor._extended` builds
  `full_k = [ek, k_img*γ_t, src_k*γ_s]`, `full_v = [ev, v_img, src_v]`,
  `full_q = [eq, q_img]`. Prompt key weight `γ_p = 1` (paper). γ multiplies the
  **target** keys; source keys held at `γ_s = 1` (`ADDIT_GAMMA_SOURCE`).
- Source K,V come from a parallel forward pass on the (re-noised) source latent,
  cached per layer: `_cache_source_kv` → `state.kv_cache[layer]`.

### §3.2 / §5 Auto-γ root-solver
- Paper: define `f(γ) = A_source − A_target` (prompt-token attention mass over
  source vs target) and root-solve `f(γ)=0`; validated default γ≈1.05.
- Code: `solve_gamma` (bisection) in `addit_core.py`; driven by
  `AddItPipeline._solve_gamma_now`, which probes single-cond forwards at trial γ
  and reads `A_source`/`A_target` accumulated in `AddItJointAttnProcessor._record_probes`.
- `ADDIT_GAMMA_MODE="auto"` re-solves every `ADDIT_GAMMA_SOLVE_EVERY` steps;
  `"fixed"` uses `ADDIT_GAMMA_FIXED=1.05` (the paper's validated constant) with
  no probing.
- **Note**: the paper computes the balance from the prompt-token attention
  distribution; we read it from the explicit softmax matrix (only when probing
  or capturing the subject map — otherwise the fused SDPA kernel is used for
  speed/memory).

### §3.2 / App A.1 Extended-attention application window
- Paper: extended attention applied in multi-stream blocks until **t=670** and
  single-stream blocks until **t=340** (out of 1000), all 30 steps.
- **SD3.5-ADAPT**: no single/multi split. We gate by remaining-noise:
  inject while `σ_t ≥ ADDIT_ATTENTION_SIGMA_MIN` (default 0.34 ≈ 340/1000),
  i.e. "apply early/mid, release late" — the paper's intent. Which *layers*
  participate is `ADDIT_ATTENTION_LAYER_FRAC` (default all 24).

### §3.3 Structure transfer
- Paper: start denoising from the source noised to a **fixed high** `t_struct`
  (`X_t = (1−σ_t)x_0 + σ_t ε`), `t_struct = 933` generated / **867** real.
- Code: `AddItPipeline.add_object` picks the start index via
  `nearest_timestep_index(timesteps, t_struct_frac·num_train)` and seeds the
  loop with `renoise_source(source_latent, noise, scheduler, timesteps[0])`.
- **SD3.5-ADAPT**: `t_struct` given as a fraction of `num_train_timesteps`
  (`ADDIT_T_STRUCT_FRAC_REAL=0.867`, `_GEN=0.933`) since SD3.5's
  FlowMatchEuler timesteps also live in `[0,1000]`, so the fractions equal the
  paper's integers.

### §3.4 Subject-guided latent blending
- Paper: gather target-patch attention to the subject token across selected
  layers/timesteps → **Otsu** rough mask `M_r` → sample ≤4 local-maxima points
  (stop below `0.35·p_max`) → **SAM-2** refine → blend at **t_blend=500**:
  `Z = M·Z_target + (1−M)·Z_source`.
- Code:
  - subject-token localisation: `AddItPipeline._subject_token_indices`
    (CLIP tokenizer; positions feed `state.subject_token_ids`),
  - attention capture: `AddItJointAttnProcessor._record_probes` accumulates the
    subject-token rows over image patches into `state.subject_attn_accum`,
    over the window `[ADDIT_SUBJECT_CAPTURE_START_FRAC, _END_FRAC]` and the
    `state.mask_layers`,
  - Otsu: `otsu_threshold` / `coarse_mask_from_attention`,
  - points: `sample_attention_points` (`max_points=4`, `rel_thresh=0.35`,
    neighborhood exclusion),
  - SAM-2: `addit_sam.refine_mask_with_sam2`,
  - blend: `latent_blend` at `i == blend_idx`
    (`blend_idx` ≈ `ADDIT_T_BLEND_FRAC·num_train`).
- **SD3.5-ADAPT (mask layers)**: the paper's FLUX mask layers
  `[13, 14, 18, single-23, single-33]` are at relative depths ≈0.6–0.95 of their
  stacks; mapped onto SD3.5's 24 blocks → `ADDIT_MASK_LAYER_FRACS =
  (0.55,0.60,0.68,0.75,0.90)`.
- **SAM-2 input image**: paper estimates `X_0` from the velocity prediction
  before SAM-2. At `t_blend≈0.5` the denoised latent is already near-clean, so
  `_build_subject_mask` decodes the current latent as the `X_0` estimate fed to
  SAM-2. (Equivalent target; avoids a separate velocity-to-X0 closed form that
  differs between FLUX and SD3.5 scheduler conventions.)

### §3.5 Real images & step-by-step
- Paper: no inversion. One random ε; each step
  `X^t_source = (1−σ_t)X_source + σ_t ε` → exact reconstruction at σ_0=0. Target
  denoises in the same batch, pulling K,V from source.
- Code: `renoise_source` is called every step to build both the cached-source
  input and the blend background; a single `noise` tensor is sampled once per
  run. `step_by_step` chains edits on the previous output.

## Deviations summary (everything not 1:1 with the paper)

1. **Backbone**: SD3.5 Medium instead of FLUX.1-dev (per repo model). Mechanism
   identical; constants re-derived as above.
2. **Guidance**: SD3.5 needs real classifier-free guidance
   (`ADDIT_GUIDANCE_SCALE`, light negative prompt). FLUX-dev used distilled
   guidance with no negative prompt. This is the one unavoidable backbone
   difference and the only place a negative prompt enters.
3. **Layer/timestep constants**: fractions, not FLUX integers (§3.2 window,
   §3.3 t_struct, §3.4 mask layers / t_blend). Defaults chosen to equal the
   paper's ratios.
4. **SAM-2**: soft dependency. If unavailable, falls back to the Otsu coarse
   mask. Paper ablation (table 7) shows coarse-mask affordance 0.809 vs full
   SAM-path 0.828 — degraded but functional.
5. **X_0 for SAM-2**: decoded current latent rather than the closed-form
   velocity→X_0 estimate (see §3.4 note). Same role, scheduler-agnostic.

## What was removed from the previous (non-faithful) implementation

The earlier `addit(experimental)/` shipped a FLUX img2img + YOLO-seg
cutout/paste augmenter that pre-decided placement with `find_insertion_region`
and ran none of the paper's mechanisms by default. All of that was removed:
no insertion bbox, no `ADDIT_VARIANT_OVERRIDES`, no YOLO person-cutout, no
`ADDIT_GENERATOR_BACKEND="flux"` img2img path. Placement is now the model's job,
which is Add-it's central claim.
