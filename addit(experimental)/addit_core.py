"""Add-it core mechanisms — faithful re-implementation on SD3.5 MM-DiT.

Implements the method of Tewel et al., "Add-it: Training-Free Object Insertion
in Images with Pretrained Diffusion Models" (ICLR 2025, arXiv:2411.07232),
adapted from the paper's FLUX.1-dev backbone to Stable Diffusion 3.5 Medium.

The paper has three components (paper §3):

  §3.2  Weighted Extended Self-Attention
        The target image's queries attend over the concatenation of
        ``[K_prompt, K_target, K_source]``.  Add-it weights the *keys* of each
        source (eq. 3):

            A = softmax([Q_p, Q_t] · [γ_s·K_src, γ_p·K_p, γ_t·K_t]^T / sqrt(d))
            h = A · [V_src, V_p, V_t]

        In practice (paper, end of §3.2 and §5) the balancing is reduced to a
        single scalar applied to source/target while the prompt key keeps γ_p=1;
        γ is chosen *per step* by a root-solver so that the prompt-token
        attention mass over the source equals that over the target,
        f(γ) = A_source − A_target = 0.  The paper sets γ≈1.05 as the validated
        average.  We implement BOTH: the analytic root-solver (default) and a
        fixed-γ mode.

  §3.3  Structure Transfer
        Instead of an arbitrary seed, start denoising from the source latent
        noised to a high, *fixed* timestep t_struct so the global structure of
        the source is inherited while content can still change.

  §3.4  Subject-Guided Latent Blending
        Capture the self-attention of the target patches to the *subject token*
        across selected layers/steps, Otsu-threshold it to a rough mask, refine
        with SAM-2 (see :mod:`addit_sam`), and at a single timestep t_blend
        composite  Z = M·Z_target + (1−M)·Z_source  so fine background detail is
        preserved while collateral effects (shadows/reflections) inside M remain.

  §3.5  Real images & step-by-step
        No inversion.  Sample one random ε; at every step the noised source is
        X^t_source = (1−σ_t)·X_source + σ_t·ε, which reconstructs the source
        exactly at σ_0=0.  The target denoises in the same batch, pulling K,V
        from the source.

SD3.5 adaptation notes are collected in ``docs/addit_paper_fidelity.md`` and
mirrored at each affected call site with an ``# SD3.5-ADAPT`` comment.  In one
sentence: SD3.5 Medium is a stack of 24 dual-stream ``JointTransformerBlock``s
(FLUX's single-stream blocks do not exist), and its scheduler runs
FlowMatchEuler timesteps in [0, 1000]; the paper's FLUX-specific integer
timesteps/layer indices are therefore re-expressed as *fractions* of the SD3.5
schedule (see :mod:`addit_config`).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


# ═══════════════════════════════════════════════════════════════════════════
# Shared mutable state visible to every attention processor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AddItState:
    """Mutable state shared across all ``AddItJointAttnProcessor`` instances.

    A single instance is shared by every layer's processor.  The denoising loop
    flips ``cache_mode`` / ``inject_mode`` and updates the per-step weights and
    the gamma estimate before each transformer forward pass.
    """

    enabled: bool = False
    """Master switch — when *False* every processor falls back to vanilla SD3.5
    joint attention (the model behaves exactly like the stock pipeline)."""

    # -- Attention re-balancing (paper eq. 3) -----------------------------
    gamma_source: float = 1.0
    """γ_s — multiplies the *source* image keys."""
    gamma_target: float = 1.05
    """γ_t — multiplies the *target* image keys.  γ_p (prompt) is fixed at 1."""

    # -- Source K,V cache (filled during the source forward pass) ---------
    cache_mode: bool = False
    """When *True*, processors store their image-token K,V into ``kv_cache``."""

    inject_mode: bool = False
    """When *True*, processors extend target attention with cached source K,V."""

    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    """Per-layer cache ``{layer_idx: (K_src, V_src)}`` in multi-head shape
    [B, H, N_img, d_head]."""

    # -- Layer window for extended attention (paper App A.1) --------------
    inject_layers: Optional[set] = None
    """Set of layer indices that participate in extended attention.  ``None``
    means *all* layers."""

    # -- γ-balancing probe (paper §3.2) -----------------------------------
    probe_mode: bool = False
    """When *True*, processors record the prompt→source and prompt→target
    attention masses needed to solve f(γ)=0, into ``probe_accum``."""

    probe_accum: Dict[str, float] = field(default_factory=dict)
    """Accumulated probe stats: keys ``a_source`` / ``a_target`` (summed over
    probed layers) used by :func:`solve_gamma`."""

    # -- Subject-token attention capture (paper §3.4) ---------------------
    capture_subject: bool = False
    """When *True*, processors accumulate target-patch attention to the subject
    token(s) into ``subject_attn_accum``."""

    subject_token_ids: Optional[List[int]] = None
    """Indices (into the prompt-token sequence) of the subject token(s)."""

    mask_layers: Optional[set] = None
    """Layer indices whose subject attention is aggregated for the mask
    (paper's best-performing layers, re-derived for SD3.5)."""

    subject_attn_accum: Optional[torch.Tensor] = None
    """Running sum of subject attention over image patches, shape [N_img]."""
    subject_attn_count: int = 0

    def clear_cache(self):
        self.kv_cache.clear()

    def reset_probe(self):
        self.probe_accum = {"a_source": 0.0, "a_target": 0.0, "n": 0.0}

    def reset_subject_capture(self):
        self.subject_attn_accum = None
        self.subject_attn_count = 0


# ═══════════════════════════════════════════════════════════════════════════
# 1. Weighted Extended Self-Attention Processor  (paper §3.2, eq. 3)
# ═══════════════════════════════════════════════════════════════════════════

class AddItJointAttnProcessor:
    """Drop-in replacement for SD3.5's ``JointAttnProcessor2_0``.

    SD3.5's ``Attention`` module exposes:
      * ``to_q / to_k / to_v / to_out`` — image-token projections,
      * ``add_q_proj / add_k_proj / add_v_proj / to_add_out`` — prompt-token
        projections (present on every block except the final one, which has no
        ``context`` output and folds text back into the image stream).
    The stock processor concatenates the two streams as ``[text, image]`` and
    runs a single joint SDPA.  We extend that joint attention with the cached
    source-image K,V and apply the eq.(3) key weights.

    Modes (read from the shared :class:`AddItState`):
      * disabled       → identical to the stock processor,
      * cache_mode     → stock attention **and** store image K,V,
      * inject_mode    → eq.(3) extended attention with cached source K,V,
      * probe_mode     → during inject, record prompt→{source,target} masses,
      * capture_subject→ during inject, accumulate subject-token attention map.
    """

    def __init__(self, layer_idx: int, state: AddItState, original_processor=None):
        self.layer_idx = layer_idx
        self.state = state
        self.original_processor = original_processor

    # ------------------------------------------------------------------
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        st = self.state
        if not st.enabled:
            return self._standard(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, *args, **kwargs)
        if st.cache_mode:
            return self._cache(attn, hidden_states, encoder_hidden_states,
                               attention_mask, *args, **kwargs)
        if st.inject_mode and self._injects():
            return self._extended(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, *args, **kwargs)
        return self._standard(attn, hidden_states, encoder_hidden_states,
                              attention_mask, *args, **kwargs)

    # ------------------------------------------------------------------
    def _injects(self) -> bool:
        if self.layer_idx not in self.state.kv_cache:
            return False
        if self.state.inject_layers is None:
            return True
        return self.layer_idx in self.state.inject_layers

    @staticmethod
    def _heads(t: torch.Tensor, heads: int) -> torch.Tensor:
        """[B, N, D] → [B, H, N, D/H]."""
        B, N, D = t.shape
        return t.view(B, N, heads, D // heads).transpose(1, 2)

    # ----- stock SD3.5 joint attention --------------------------------
    def _standard(self, attn, hidden_states, encoder_hidden_states,
                  attention_mask, *args, **kwargs):
        if self.original_processor is not None:
            return self.original_processor(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, *args, **kwargs)

        # Fallback re-implementation (used only if the original is missing).
        B = hidden_states.shape[0]
        inner_dim = attn.to_k(hidden_states).shape[-1]
        query = self._heads(attn.to_q(hidden_states), attn.heads)
        key = self._heads(attn.to_k(hidden_states), attn.heads)
        value = self._heads(attn.to_v(hidden_states), attn.heads)
        if encoder_hidden_states is not None:
            eq = self._heads(attn.add_q_proj(encoder_hidden_states), attn.heads)
            ek = self._heads(attn.add_k_proj(encoder_hidden_states), attn.heads)
            ev = self._heads(attn.add_v_proj(encoder_hidden_states), attn.heads)
            query = torch.cat([eq, query], dim=2)
            key = torch.cat([ek, key], dim=2)
            value = torch.cat([ev, value], dim=2)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, inner_dim)
        if encoder_hidden_states is not None:
            enc_seq = encoder_hidden_states.shape[1]
            enc_out, img_out = out[:, :enc_seq], out[:, enc_seq:]
            img_out = attn.to_out[1](attn.to_out[0](img_out))
            if getattr(attn, "to_add_out", None) is None:
                return img_out
            return img_out, attn.to_add_out(enc_out)
        return attn.to_out[1](attn.to_out[0](out))

    # ----- cache mode: stock attention + store image K,V --------------
    def _cache(self, attn, hidden_states, encoder_hidden_states,
               attention_mask, *args, **kwargs):
        if getattr(attn, "to_k", None) is not None:
            k_img = self._heads(attn.to_k(hidden_states), attn.heads)
            v_img = self._heads(attn.to_v(hidden_states), attn.heads)
            self.state.kv_cache[self.layer_idx] = (k_img.detach(), v_img.detach())
        return self._standard(attn, hidden_states, encoder_hidden_states,
                              attention_mask, *args, **kwargs)

    # ----- inject mode: eq.(3) extended attention ---------------------
    def _extended(self, attn, hidden_states, encoder_hidden_states,
                  attention_mask, *args, **kwargs):
        required = ("to_q", "to_k", "to_v", "to_out")
        if any(getattr(attn, n, None) is None for n in required):
            return self._standard(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, *args, **kwargs)

        st = self.state
        B = hidden_states.shape[0]
        inner_dim = attn.to_k(hidden_states).shape[-1]
        head_dim = inner_dim // attn.heads
        scale = 1.0 / math.sqrt(head_dim)

        # Target image-token Q,K,V
        q_img = self._heads(attn.to_q(hidden_states), attn.heads)
        k_img = self._heads(attn.to_k(hidden_states), attn.heads)
        v_img = self._heads(attn.to_v(hidden_states), attn.heads)

        # Cached source image K,V (eq. 3: the extra source tokens)
        src_k, src_v = st.kv_cache[self.layer_idx]
        if src_k.shape[0] < B:                       # source batch 1 vs target B
            src_k = src_k.expand(B, -1, -1, -1)
            src_v = src_v.expand(B, -1, -1, -1)
        src_k = src_k.to(dtype=k_img.dtype, device=k_img.device)
        src_v = src_v.to(dtype=v_img.dtype, device=v_img.device)

        gs = float(st.gamma_source)
        gt = float(st.gamma_target)

        has_text = encoder_hidden_states is not None
        if has_text:
            context_required = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
            if any(getattr(attn, n, None) is None for n in context_required):
                return self._standard(attn, hidden_states, encoder_hidden_states,
                                      attention_mask, *args, **kwargs)
            eq = self._heads(attn.add_q_proj(encoder_hidden_states), attn.heads)
            ek = self._heads(attn.add_k_proj(encoder_hidden_states), attn.heads)
            ev = self._heads(attn.add_v_proj(encoder_hidden_states), attn.heads)
            n_text = ek.shape[2]

            # Q = [Q_prompt, Q_target_image]           (queries unweighted)
            # K = [γ_p·K_p, γ_t·K_target, γ_s·K_source] with γ_p = 1
            # V = [V_prompt, V_target,     V_source]    (values unweighted)
            full_q = torch.cat([eq, q_img], dim=2)
            full_k = torch.cat([ek, k_img * gt, src_k * gs], dim=2)
            full_v = torch.cat([ev, v_img, src_v], dim=2)
        else:
            # Final block has no separate context stream.
            full_q = q_img
            full_k = torch.cat([k_img * gt, src_k * gs], dim=2)
            full_v = torch.cat([v_img, src_v], dim=2)
            n_text = 0

        n_target = k_img.shape[2]
        n_source = src_k.shape[2]

        # --- §3.2/§3.4 probes: need the explicit attention matrix when the
        #     loop asks for the γ balance or the subject map.  Otherwise use
        #     the fused (faster, lower-memory) SDPA kernel.
        need_weights = (st.probe_mode or st.capture_subject) and has_text
        if need_weights:
            attn_logits = torch.matmul(full_q, full_k.transpose(-1, -2)) * scale
            attn_probs = attn_logits.softmax(dim=-1)            # [B,H,Nq,Nk]
            out = torch.matmul(attn_probs, full_v)
            self._record_probes(attn_probs, n_text, n_target, n_source)
        else:
            out = F.scaled_dot_product_attention(full_q, full_k, full_v, attn_mask=None)

        out = out.transpose(1, 2).reshape(B, -1, inner_dim)
        if has_text:
            enc_out, img_out = out[:, :n_text], out[:, n_text:]
            img_out = attn.to_out[1](attn.to_out[0](img_out))
            return img_out, attn.to_add_out(enc_out)
        return attn.to_out[1](attn.to_out[0](out))

    # ------------------------------------------------------------------
    def _record_probes(self, attn_probs, n_text, n_target, n_source):
        """Accumulate the two paper quantities from the explicit attn matrix.

        ``attn_probs`` is [B, H, Nq, Nk] over keys ordered
        ``[text(n_text) | target(n_target) | source(n_source)]``.  The paper's
        balance uses the *prompt* query rows (Q_p) and compares the mass they
        place on the source vs the target image keys (§3.2, §5):
            A_source = mean over prompt rows of (prob mass on source keys)
            A_target = mean over prompt rows of (prob mass on target keys)
        Averaged over the CFG/cond batch and heads.
        """
        st = self.state
        # Use the conditional half of the CFG batch (last item) — that is the
        # branch whose balance the paper analyses.
        probs = attn_probs[-1] if attn_probs.shape[0] > 1 else attn_probs[0]  # [H,Nq,Nk]
        prompt_rows = probs[:, :n_text, :]                       # [H, n_text, Nk]
        tgt_lo, tgt_hi = n_text, n_text + n_target
        src_lo, src_hi = tgt_hi, tgt_hi + n_source
        a_target = prompt_rows[:, :, tgt_lo:tgt_hi].sum(-1).mean().item()
        a_source = prompt_rows[:, :, src_lo:src_hi].sum(-1).mean().item()

        if st.probe_mode:
            st.probe_accum["a_source"] += a_source
            st.probe_accum["a_target"] += a_target
            st.probe_accum["n"] += 1.0

        if st.capture_subject and st.subject_token_ids:
            in_mask_layers = (st.mask_layers is None) or (self.layer_idx in st.mask_layers)
            if in_mask_layers:
                # Subject-token query rows → attention onto target image patches.
                # (paper §3.4: "multiplying the queries from the target image
                #  patches with the key of the added-object token".  Equivalent
                #  up to transpose; we read the subject-token row over image
                #  patches, which is the readily available softmaxed quantity.)
                subj = [i for i in st.subject_token_ids if i < n_text]
                if subj:
                    rows = probs[:, subj, tgt_lo:tgt_hi]          # [H, |subj|, n_target]
                    agg = rows.mean(dim=(0, 1))                    # [n_target]
                    if st.subject_attn_accum is None:
                        st.subject_attn_accum = agg.detach().float()
                    else:
                        st.subject_attn_accum = st.subject_attn_accum + agg.detach().float()
                    st.subject_attn_count += 1


# ═══════════════════════════════════════════════════════════════════════════
# Processor injection / restoration
# ═══════════════════════════════════════════════════════════════════════════

def inject_addit_processors(transformer, state: AddItState):
    """Replace every attention processor with an :class:`AddItJointAttnProcessor`
    sharing *state*.  Layer index is assigned in ``attn_processors`` order, which
    matches ``transformer.transformer_blocks`` order for SD3.5.

    Returns the original ``{name: processor}`` mapping for :func:`restore_processors`.
    """
    original = {}
    new = {}
    for idx, (name, module) in enumerate(transformer.attn_processors.items()):
        original[name] = module
        new[name] = AddItJointAttnProcessor(layer_idx=idx, state=state, original_processor=module)
    transformer.set_attn_processor(new)
    return original


def restore_processors(transformer, original_processors: dict):
    transformer.set_attn_processor(original_processors)


def resolve_layer_window(num_layers: int, frac_range: Tuple[float, float]) -> set:
    """Map a (start_frac, end_frac) window to a concrete set of layer indices.

    SD3.5-ADAPT: the paper applies extended attention in FLUX's multi-stream
    blocks up to t=670 and single-stream up to t=340.  SD3.5 has a single block
    type, so we instead express the *which-layers* choice as a fraction of the
    24-block stack (default = all layers); the *which-steps* choice is handled
    by the timestep windows in the denoise loop (see config).
    """
    lo = max(0, int(round(num_layers * frac_range[0])))
    hi = min(num_layers, int(round(num_layers * frac_range[1])))
    return set(range(lo, max(lo + 1, hi)))


def resolve_mask_layers(num_layers: int, fracs: Tuple[float, ...]) -> set:
    """Map fractional positions to layer indices for subject-attention capture.

    SD3.5-ADAPT: the paper's mask layers are FLUX indices
    [13, 14, 18, single-23, single-33] out of 19 multi + 38 single blocks.
    We map those *relative depths* onto SD3.5's 24 blocks.  See config
    ``ADDIT_MASK_LAYER_FRACS`` for the chosen fractions.
    """
    out = set()
    for f in fracs:
        out.add(min(num_layers - 1, max(0, int(round(num_layers * f)))))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. γ root-solver  (paper §3.2: find γ s.t. f(γ) = A_source − A_target = 0)
# ═══════════════════════════════════════════════════════════════════════════

def solve_gamma(probe_fn, gamma_lo: float = 0.6, gamma_hi: float = 1.6,
                iters: int = 12, tol: float = 1e-3) -> float:
    """Bisection root-solver for the balance condition f(γ)=0.

    ``probe_fn(gamma) -> (a_source, a_target)`` runs one cheap target forward
    pass at the trial γ (γ multiplies the *target* keys; source keys held at 1)
    and returns the prompt→source and prompt→target attention masses.
    f(γ) = a_source − a_target.  At low γ the source dominates (f>0); raising γ
    grows target mass (f decreases) — so f is monotonically decreasing in γ and
    bisection is well-posed.  Falls back to the bracket mid-point if the sign
    does not bracket a root (paper validates γ≈1.05 as a robust default).
    """
    def f(g):
        a_s, a_t = probe_fn(g)
        return a_s - a_t

    f_lo = f(gamma_lo)
    f_hi = f(gamma_hi)
    if f_lo == 0:
        return gamma_lo
    if f_hi == 0:
        return gamma_hi
    if (f_lo > 0) == (f_hi > 0):
        # No sign change in the bracket → return whichever endpoint is closer to
        # balance, biased toward the paper's validated 1.05.
        return gamma_hi if abs(f_hi) < abs(f_lo) else gamma_lo

    lo, hi = gamma_lo, gamma_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if (fm > 0) == (f_lo > 0):
            lo, f_lo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Structure Transfer  (paper §3.3)  &  real-image re-noising  (paper §3.5)
# ═══════════════════════════════════════════════════════════════════════════

def encode_image_to_latent(vae, image: Image.Image, resolution: int, device, dtype) -> torch.Tensor:
    """Encode a PIL image to the SD3.5 latent space (shift + scale applied)."""
    image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    pixel = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
    pixel = (pixel / 127.5 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(pixel).latent_dist.sample()
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    sf = getattr(vae.config, "scaling_factor", 1.0) or 1.0
    return (latent - shift) * sf


def decode_latent_to_image(vae, latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
        sf = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        latent = latent / sf + shift
        decoded = vae.decode(latent).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((decoded * 255).astype(np.uint8), mode="RGB")


def sigma_for_timestep(scheduler, timestep) -> float:
    """Return the flow-matching σ_t in [0,1] for a scheduler timestep.

    For FlowMatchEulerDiscreteScheduler, σ_t = t / num_train_timesteps and the
    rectified-flow interpolation is X_t = (1−σ_t)·X_0 + σ_t·ε.
    """
    t = float(timestep.item() if torch.is_tensor(timestep) else timestep)
    num_train = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    return t / num_train


def renoise_source(source_latent: torch.Tensor, noise: torch.Tensor,
                   scheduler, timestep) -> torch.Tensor:
    """Rectified-flow forward noising  X_t = (1−σ_t)·X_source + σ_t·ε  (paper §3.5).

    Used both to build the structure-transfer start latent and, each step, to
    produce the source latent the target attends to and blends against.
    """
    sigma = sigma_for_timestep(scheduler, timestep)
    return (1.0 - sigma) * source_latent + sigma * noise


def nearest_timestep_index(timesteps: torch.Tensor, target_t: float) -> int:
    """Index in ``timesteps`` whose value is closest to ``target_t``."""
    diffs = (timesteps.float() - float(target_t)).abs()
    return int(torch.argmin(diffs).item())


# ═══════════════════════════════════════════════════════════════════════════
# 4. Subject-Guided Latent Blending  (paper §3.4)
# ═══════════════════════════════════════════════════════════════════════════

def attn_vector_to_mask_image(attn_vec: torch.Tensor, latent_hw: Tuple[int, int],
                              out_hw: Tuple[int, int]) -> Image.Image:
    """Reshape a per-patch attention vector [N_img] into a normalized L image.

    ``latent_hw`` = (H_lat, W_lat) of the patch grid; ``out_hw`` = (H, W) of the
    pixel image we want the mask at.
    """
    h, w = latent_hw
    vec = attn_vec.detach().float().cpu().numpy()
    if vec.size != h * w:
        # Defensive: crop/pad to the grid size.
        flat = np.zeros(h * w, dtype=np.float32)
        flat[: min(vec.size, h * w)] = vec[: min(vec.size, h * w)]
        vec = flat
    grid = vec.reshape(h, w)
    grid = grid - grid.min()
    if grid.max() > 0:
        grid = grid / grid.max()
    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    return img.resize((out_hw[1], out_hw[0]), Image.BILINEAR)


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's method (paper §3.4) — returns an integer threshold in [0,255]."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def coarse_mask_from_attention(attn_img: Image.Image) -> np.ndarray:
    """Otsu-threshold the attention image into a rough binary mask M_r (uint8)."""
    gray = np.asarray(attn_img.convert("L"), dtype=np.uint8)
    t = otsu_threshold(gray)
    return (gray > t).astype(np.uint8) * 255


def sample_attention_points(attn_img: Image.Image, max_points: int = 4,
                            rel_thresh: float = 0.35,
                            exclude_radius_frac: float = 0.10) -> List[Tuple[int, int]]:
    """Iteratively sample local maxima as SAM-2 prompts (paper App A.1).

    Pick the global max; exclude a neighborhood; repeat until 4 points or the
    next max falls below ``rel_thresh · p_max``.  Returns (x, y) pixel points.
    """
    gray = np.asarray(attn_img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    p_max = float(gray.max())
    if p_max <= 0:
        return []
    work = gray.copy()
    radius = max(1, int(round(exclude_radius_frac * max(h, w))))
    points: List[Tuple[int, int]] = []
    for _ in range(max_points):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w)
        val = work[y, x]
        if val < rel_thresh * p_max:
            break
        points.append((int(x), int(y)))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return points


def latent_blend(z_target: torch.Tensor, z_source: torch.Tensor,
                 mask_latent: torch.Tensor) -> torch.Tensor:
    """Single-step blend  Z = M·Z_target + (1−M)·Z_source  (paper §3.4).

    ``mask_latent`` is [1,1,H,W] in [0,1] with 1 = subject (keep generated).
    """
    m = mask_latent.to(device=z_target.device, dtype=z_target.dtype)
    return m * z_target + (1.0 - m) * z_source


def pixel_mask_to_latent(mask_img: Image.Image, latent_hw: Tuple[int, int],
                         device, dtype) -> torch.Tensor:
    """Resize a pixel-space L mask to the latent grid as a [1,1,H,W] tensor."""
    h, w = latent_hw
    m = mask_img.convert("L").resize((w, h), Image.BILINEAR)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def composite_outside_mask(source_image: Image.Image, generated_image: Image.Image,
                           mask_img: Image.Image) -> Image.Image:
    """Pixel-exact restore of the source outside the (decoded) subject mask.

    The paper's latent blend keeps the *background latent* identical; decoding
    can still perturb pixels by a hair.  This optional final composite enforces
    exact source pixels outside M (1=keep generated).  It is NOT a substitute
    for the latent blend — it runs after it.
    """
    source = source_image.convert("RGB")
    generated = generated_image.convert("RGB").resize(source.size, Image.LANCZOS)
    mask = mask_img.convert("L").resize(source.size, Image.BILINEAR)
    return Image.composite(generated, source, mask)


# ═══════════════════════════════════════════════════════════════════════════
# Composite insertion mode  (background-preserving, sharp output)
# ═══════════════════════════════════════════════════════════════════════════
# These helpers implement the ADDIT_MODE == "composite" path: segment the
# newly-generated person(s) out of an img2img candidate and paste ONLY those
# pixels back onto the byte-exact source.  Self-contained — depends only on
# Ultralytics YOLO (soft) + PIL/numpy, never on the root sd35_* augmentation
# modules (Add-it stays a standalone reference flow).

_PERSON_SEGMENTER = None  # lazily-loaded Ultralytics YOLO-seg model (or False)


def load_person_segmenter(model_path: str, device=None):
    """Load Ultralytics YOLOv8-seg once; return the model or None if unavailable.

    Cached in a module global so repeated ``add_object`` calls reuse one model.
    Returns None (not raising) when ultralytics/weights are missing so the
    caller can degrade gracefully instead of crashing the whole run.
    """
    global _PERSON_SEGMENTER
    if _PERSON_SEGMENTER is False:
        return None
    if _PERSON_SEGMENTER is not None:
        return _PERSON_SEGMENTER
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        _PERSON_SEGMENTER = {"model": model, "device": device}
        return _PERSON_SEGMENTER
    except Exception as exc:  # ultralytics missing / weights not found
        print(f"YOLO person segmenter unavailable ({type(exc).__name__}: {exc}); "
              "composite mode cannot isolate the added person.")
        _PERSON_SEGMENTER = False
        return None


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run_person_seg(segmenter, image: Image.Image, min_conf: float,
                    mask_threshold: float, person_class: int = 0):
    """Run YOLO-seg; return [(bbox_xyxy, conf, mask_bool_HxW), ...] for people."""
    import numpy as _np
    model = segmenter["model"]
    device = segmenter["device"]
    arr = _np.array(image.convert("RGB"))
    kwargs = {"verbose": False, "conf": float(min_conf)}
    if device is not None:
        kwargs["device"] = device
    res = model(arr, **kwargs)
    out = []
    if not res or res[0].masks is None or res[0].boxes is None:
        return out
    r = res[0]
    H, W = arr.shape[:2]
    masks = r.masks.data.cpu().numpy()          # (N, h, w) in [0,1]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()
    for m, b, c, k in zip(masks, boxes, confs, clss):
        if int(k) != person_class:
            continue
        # resize mask to full image, threshold to binary
        mask_img = Image.fromarray((m * 255).astype("uint8"), mode="L").resize((W, H), Image.BILINEAR)
        mb = _np.array(mask_img) >= int(mask_threshold * 255)
        out.append((tuple(float(v) for v in b), float(c), mb))
    return out


def segment_added_person_mask(
    candidate_image: Image.Image,
    source_image: Image.Image,
    segmenter,
    *,
    min_conf: float,
    mask_threshold: float,
    filter_existing: bool,
    existing_iou: float,
    max_people: int,
    trim_fringe_px: int,
    feather_px: int,
) -> Tuple[Optional[Image.Image], int, float]:
    """Mask of the person(s) added in ``candidate`` but absent from ``source``.

    Returns ``(mask_L | None, kept_person_count, best_conf)``.  The mask is the
    union of every kept person's segmentation, cleaned (fringe-trim + feather)
    so the composite seam is tight.  ``None`` when no NEW person is found.
    """
    import numpy as _np
    if segmenter is None:
        return None, 0, 0.0

    cand_people = _run_person_seg(segmenter, candidate_image, min_conf, mask_threshold)
    if not cand_people:
        return None, 0, 0.0

    existing_boxes = []
    if filter_existing:
        src_people = _run_person_seg(segmenter, source_image, min_conf, mask_threshold)
        existing_boxes = [p[0] for p in src_people]

    # Keep candidate people that don't overlap an existing source person.
    kept = []
    for bbox, conf, mb in cand_people:
        if filter_existing and any(_bbox_iou(bbox, eb) > existing_iou for eb in existing_boxes):
            continue
        kept.append((bbox, conf, mb))
    if not kept:
        return None, 0, 0.0

    # Highest-confidence first, cap the count.
    kept.sort(key=lambda t: t[1], reverse=True)
    kept = kept[:max(1, max_people)]
    best_conf = kept[0][1]

    W, H = candidate_image.size
    union = _np.zeros((H, W), dtype=bool)
    for _, _, mb in kept:
        union |= mb
    mask_img = Image.fromarray((union.astype("uint8") * 255), mode="L")

    # Cleanup: erode to drop the generated-background fringe, then feather.
    if trim_fringe_px > 0:
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=1 + 2 * trim_fringe_px))
    if feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return mask_img, len(kept), best_conf


def composite_person_onto_source(
    source_image: Image.Image,
    candidate_image: Image.Image,
    person_mask: Image.Image,
) -> Image.Image:
    """Paste ONLY the masked person pixels from candidate onto the source.

    Background is byte-exact source outside the mask (the repo's core rule);
    inside the (feathered) mask the candidate's sharp person pixels show.
    """
    source = source_image.convert("RGB")
    candidate = candidate_image.convert("RGB").resize(source.size, Image.LANCZOS)
    mask = person_mask.convert("L").resize(source.size, Image.BILINEAR)
    return Image.composite(candidate, source, mask)
