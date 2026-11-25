# TDA-Based Defense Against Data Poisoning in Network Intrusion Detection

This repository contains a research pipeline for identifying and mitigating poisoned samples in network intrusion detection data. It combines payload processing, topological data analysis (TDA), clustering-based detection, dataset sanitization, and a deep-learning classifier.

## Repository contents

- `src/` — pipeline stages and the final NIDS training script
- `configs/` — experiment settings
- `data/sanitized/cleaned_dataset_new.csv` — sanitized dataset supplied with this project
- `results/` — evaluation metrics, tables, plots, and training history
- `artifacts/` — saved classifier and preprocessing object
- `*_summary*.md`, `project_documentation.md`, `presentation_slides.md`, and `reproduction_research_paper.md` — project documentation

The virtual environment and large raw/intermediate datasets are intentionally not included. The sanitized CSV is the same content the final training script previously loaded as `cleaned_dataset.csv`.

## Run the final classifier

Use Python 3.10 or newer in a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/10_deep_learning_nids.py
```

The final script reads `data/sanitized/cleaned_dataset_new.csv` and writes metrics and training history under `results/`, plus the trained model and preprocessor under `artifacts/`.

## Reproduce the full pipeline

The earlier pipeline stages require raw and intermediate data files that are not bundled in this repository. Obtain the source dataset separately, place the required files in the paths specified in `configs/reproduction_config.yaml`, and then run stages `01` through `10` from the repository root. The included sanitized CSV is sufficient to run stage `10` directly.

## Dependencies

Dependencies are listed in `requirements.txt`. TensorFlow is needed for the final classifier; `giotto-tda` and `hdbscan` are used by the TDA and clustering stages.
