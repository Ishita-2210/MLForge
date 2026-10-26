# Repository preparation

Suggested GitHub repository name: **MLForge**.

Description: **Local CPU-first tabular ML workbench with leakage-safe experiments, Streamlit, FastAPI, SQLite tracking, a versioned model registry and input-drift diagnostics.**

Suggested topics: `machine-learning`, `scikit-learn`, `mlops`, `streamlit`, `fastapi`, `python`, `sqlite`, `model-registry`, `tabular-data`, `data-drift`.

This is a standalone local Git repository using the configured user identity. The CI workflow defines actual lint, test, import and reproducible smoke commands on supported Python/OS combinations, with no hardware or cloud-service requirement. Local checks were executed; hosted CI has not run because this project has no published remote. No passing CI badge or live deployment URL is invented.

Source inspection covers `.gitignore`, diffs, generated/private data exclusion, sample provenance and working README commands. Only publish the source/docs/CI and small genuine UI screenshots. Do not push `.venv`, `.mlforge`, SQLite files, datasets, joblib artifacts, logs or build data. Do not attach the TinyLink/Image Entropy repository's remote or force-push over existing history.

The portfolio was inspected and left unchanged: adding an accurate project entry should use an actual published repository URL after publication, rather than an invented link. No paid deployment is needed; the demonstrated local web UI is sufficient.
