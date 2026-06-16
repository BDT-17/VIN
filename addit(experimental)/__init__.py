"""Add-it: Training-Free Object Insertion for CityPersons Augmentation.

Implements the Add-it architecture (arXiv 2411.07232) adapted for SD3.5 Medium:
  1. Weighted Extended-Attention — inject source scene K,V into transformer attention
  2. Noise Structure Transfer — start denoising from source-structured latent
  3. Subject-Guided Latent Blending — preserve background via per-step latent masking
"""
