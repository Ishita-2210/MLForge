# Executed validation — 27 October 2026

## Actual environment and test scope

Windows 11, Intel i5-1135G7 (4 physical cores/8 logical processors), approximately 16 GB RAM; Python 3.13.3, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.2.6, SciPy 1.15.3, joblib 1.4.2. All training/inference used CPU. No external dataset or paid service was involved.

**66 automated tests passed, zero failed.** The deployment validation run took **48.93 seconds**. Two third-party deprecation warnings remain (Starlette's AnyIO alias and Streamlit's Altair theme API); neither failed the checks. Ruff lint and format checks passed, Python compilation/import checks passed, and a package wheel built successfully during initial local validation. Four added tests cover hosted workspace isolation, path validation, session caps and the container entrypoint's security settings.

The tests cover strict CSV parsing/limits/identity, profiling, target/config validation, preprocessing fitted only on training, disjoint group/time folds and chronology/gaps, rare-class CV fallback, bounded search, all supported model families, binary/multiclass metrics, final-test isolation, partial/all candidate failures, artifact reload/integrity/compatibility/version history, prediction feature ordering, missing/unseen values, API schema/body/host safety, monitoring retention/drift, background-process persistence, cancellation/recovery and the complete Streamlit workflow.

## Reproducible synthetic results

Command executed from the project root:

```powershell
.\.venv\Scripts\python -m mlforge.cli --home build/verified-smoke smoke --task both --output build/verified-benchmark.json
```

Each generated dataset has **600 rows**, eight selected features (six numeric signals, segment and boolean), and seed **42**. Split: **360 training / 120 validation / 120 final holdout**. Candidates: baseline, linear, tree and forest, no tuning/CV. All four candidates succeeded for both tasks. The selected fitted model accessed the final holdout once, after validation-based selection. Registration, reload through a fresh service instance, two-row prediction and the synthetic drift demonstration all completed.

| Measurement | Classification | Regression |
| --- | ---: | ---: |
| Selected algorithm | Random forest | Ridge |
| Validation primary metric | Macro-F1 **0.924908** | RMSE **31.166650** |
| Final holdout accuracy | **0.883333** | Not applicable |
| Final holdout macro-F1 | **0.882441** | Not applicable |
| Final holdout ROC-AUC | **0.976250** | Not applicable |
| Final holdout MAE | Not applicable | **16.719822** |
| Final holdout RMSE | Not applicable | **23.020984** |
| Final holdout R² | Not applicable | **0.942742** |
| Training/evaluation/explainability duration | **1.550559 s** | **1.210633 s** |
| Observed service prediction duration (2 rows) | **93.3020 ms** | **14.6598 ms** |
| Compressed registered pipeline | **74233 bytes** | **2188 bytes** |
| Process RSS snapshot before training | **177012736 bytes** | **179658752 bytes** |
| Process RSS snapshot after training | **178139136 bytes** | **179929088 bytes** |

These timings are individual observed runs, not latency percentiles or a speed comparison between algorithms/tasks. They include fitted preprocessing and validation/importance work where labeled. Prediction timing includes artifact integrity/load, preprocessing, model/probabilities and drift computation; it excludes HTTP transport, queueing and final monitoring-log commit. RSS snapshots describe the process, not peak memory or isolated per-model allocation. [Machine-readable summary](benchmark.json) contains candidate fit/attempt timings and environment metadata.

Performance is on **synthetic demonstration problems**, not a real-world business benchmark. Independent held-out data from the intended domain is needed before making external performance claims. High validation/test scores do not remove domain-leakage risk.

## Browser and live API evidence

Actual Chromium verification used the running local UI/API. A **240-row synthetic CSV** was uploaded through the file widget, validated/stored/profiled, and used to review and start a real background experiment. Its selected tree, validation/final metrics, registry copy and reload were confirmed. Browser Playground predictions and a real HTTP prediction returned results from that persisted model. A deliberately incorrect feature schema returned HTTP 422. Actual monitoring entries, including a labeled artificial drift demonstration, were retrieved. No browser JavaScript errors were recorded.

This browser run is a different dataset/configuration from the 600-row benchmark table; its metrics must not be substituted for those results. Screenshots in `docs/assets/` are actual local UI captures. The empty overview capture shows an actual fresh workspace with zero counts. Python Streamlit AppTest independently reproduces the same main user workflow during ordinary tests, without requiring Node or browser installation.

## Reproduce checks

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m compileall -q app mlforge scripts streamlit_app.py
.\.venv\Scripts\python -m pytest -ra
.\.venv\Scripts\python -m mlforge.cli --home build/reproduction smoke --task both --output build/reproduction.json
```

Raw local logs/benchmark JSONs are under ignored `build/`. Generated datasets, SQLite databases and trained joblib files are intentionally excluded from source control. [GitHub Actions](https://github.com/Ishita-2210/MLForge/actions) defines equivalent checks for Ubuntu Python 3.11/3.13 and Windows Python 3.13. Consult the actual run status there. Real-data generalization, authenticated durable hosting and outcome-based performance monitoring remain outside this release.

## Production demo verification

The public production alias **https://mlforge-nine.vercel.app** returned HTTP 200 for the application and `/_stcore/health`. An actual unauthenticated Chromium session opened the WebSocket application, uploaded the same 240-row public synthetic QA CSV, profiled it, reviewed a configuration, started a background CPU experiment, received final holdout metrics, registered the fitted pipeline and made a real Playground prediction. The synthetic input-shift action returned computed PSI/total-variation diagnostics and flagged features. No browser JavaScript errors or HTTP 5xx responses occurred.

A second independent browser context showed four zero-count overview metrics, confirming it did not inherit the first visitor's datasets, experiments or registered models. This checks session separation in the deployed demo; it is not a claim of audited tenant security. The deployed workspace is temporary and the dedicated REST API remains a local option. The existing benchmark table above still reports the original local 600-row measurements, not cloud timings. Deployment QA results and screenshots are recorded under ignored `build/hosted-*`.
