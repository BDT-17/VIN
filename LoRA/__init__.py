"""VIN LoRA toolkit.

Two independent flows:

  A. Data ETL + Train  (LoRA.data, LoRA.train)
       raw datasets -> LoRA release -> train adapter -> model artifacts

  B. SD3.5 Inpaint Test  (LoRA.inference)
       frozen test images + masks -> SD3.5 inpaint baseline
       -> SD3.5 inpaint + LoRA -> paired metrics + report

This package contains NO copy of the V5 augmentation flow. There is no scale
correction, semantic placement, object-only composite, harmonization, or
autotune here. All modules use clean package-relative imports.
"""
