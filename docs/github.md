# Repository and run options

- Repository: **https://github.com/Ishita-2210/MLForge**
- Live demo / repository homepage: **https://mlforge-nine.vercel.app**
- Local project folder on the original computer: **`D:\ml forge`**
- Local UI: **http://127.0.0.1:8501**; dedicated API documentation: **http://127.0.0.1:8000/docs**
- Local setup: [README instructions](../README.md#run-on-windows).

The public GitHub repository contains source, documentation, CI and genuine UI screenshots. Runtime uploads, SQLite databases, trained artifacts, logs, build output, virtual environments and deployment credentials are excluded. The repository is separate from TinyLink AI and Image Entropy Analyzer; their remotes and histories were preserved.

The live demo was deployed and exercised before publishing this source. It performs real training and inference with temporary per-session storage. Local mode provides persistent storage and the dedicated REST API. The GitHub About homepage and README link to the deployed app, and the hosted sidebar links back to the source and local setup.

Repository topics: `machine-learning`, `scikit-learn`, `mlops`, `streamlit`, `fastapi`, `python`, `sqlite`, `model-registry`, `tabular-data`, `data-drift`.

The CI workflow runs lint, formatting, imports, tests and classification/regression smoke checks on Linux Python 3.11/3.13 and Windows Python 3.13. [View actual CI runs](https://github.com/Ishita-2210/MLForge/actions). Local and deployed verification details are in [validation.md](validation.md). The portfolio remains unchanged.
