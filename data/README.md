# Data directory

Everything under data/ except this file is gitignored and must stay local.

Layout (created automatically by `pokertell` commands):

- raw/       downloaded session videos. Never committed, never redistributed.
- frames/    sampled frames and ROI crops for OCR calibration.
- hands/     reconstructed hand histories, one file per session.
- features/  per-decision feature tables.
- models/    trained model artifacts.
- reports/   metrics tables and calibration plots.

Policy: only footage from streams where players consented to hole-card
broadcast (Hustler Casino Live), analysis is post hoc only, and no footage or
clips are ever published. What gets shared is code, derived numeric features,
and timestamps. See the Ethics and data section of the top-level README.
