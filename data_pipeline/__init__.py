"""Phase 2 data pipeline. Specialty-pluggable from day one.

Adding a specialty:
1. Create `ingest/<specialty>/<society>.py` implementing `Ingester`.
2. Register the society + grading vocabulary in `registry.py`.
3. Provide a normalization function mapping native grades onto
   `evidence_strength_score in [0, 1]`.
"""
