"""Shared VAE-encode helper for the SD3 flow-matching trainers.

Lifted out of the (removed) mask-free/inpaint spike so the concept trainer has a
neutral home for it, with no edit-flow dependency.
"""


def _vae_encode(vae, images, mode=False):
    """Encode to SD3 latent space. mode=True uses the distribution MODE
    (deterministic) — PIPE/IP2P encodes the CONDITIONING (source) image with
    .mode() so the source signal carries no sampling noise; the noisy-target
    latent keeps the default .sample()."""
    dist = vae.encode(images).latent_dist
    lat = dist.mode() if mode else dist.sample()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor
