# training/

Phase 2 (separate plan) lives here. Phase 1 only needs the directory to exist
so the `GISModel` Protocol seam in `src/weighted_evidence/rubric/gis.py`
can be replaced without touching call sites.

Recommended starting point when Phase 2 begins:

- Base model: **Llama-3.1-8B-Instruct** with LoRA via PEFT/TRL. Alt:
  BioMistral-7B.
- Multi-task objective:
  1. Predict whether a paper would be cited in a guideline of a given
     specialty.
  2. Predict the guideline-assigned `evidence_strength_score in [0, 1]`.
  3. Predict the strength of the recommendation it supports
     (Strong / Conditional / Ungraded).
- Training data: assembled by `data_pipeline/` for each registered specialty.
  Each (paper, guideline, recommendation) triple is one labeled example.
- Eval: held-out specialties for cross-specialty generalization; ablate the
  guideline-assigned-strength signal to confirm it's the dominant feature.
