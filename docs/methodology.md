# ML methodology and reproducibility

## What is selected

`ExperimentConfig` records dataset ID, task, target, ordered features, primary metric, candidates, preprocessing, random seed, validation/test proportions, random/group/time strategy and keys, time gap, requested CV, bounded search and time budget. The experiment record adds the immutable dataset fingerprint, warnings, candidate parameters/failures/times, effective CV, exact row indices, selected candidate, validation/final metrics, artifact checksum and software metadata.

Default random classification stratifies both splits. Regression uses seeded random partitions. Group splitting uses disjoint groups in train/validation/test, with GroupKFold restricted to training groups when requested. Time splitting sorts distinct timestamps, keeps equal timestamps together, and can purge the final timestamp groups before each subsequent boundary. Its expanding CV also operates on timestamp groups and respects the gap. Group sizes may produce proportions different from exact requested row percentages; actual rows are displayed. Split keys are excluded from features/target.

Rare/insufficient CV folds fall back to the independent validation holdout with a saved warning. A final split with too few rows or a training split missing a target class is rejected. At least 30 labeled rows are required; small-dataset and rare-class warnings remain visible. Ingestion retains duplicate rows; users must choose an appropriate entity grouping if related rows would leak across partitions.

## Preprocessing boundary

Type inference and quality profiling describe input data; they do not provide fitted imputations or vocabularies. Each candidate/fold constructs a fresh `ColumnTransformer`/`Pipeline`. Numeric imputation uses median or mean and retains empty columns. Numeric scaling is applied only for linear models. Categorical imputation uses most-frequent or constant filling, followed by dense one-hot encoding with at most 24 categories per column and fitted unknown handling. Deterministic type normalization and selected feature order are reused at inference. Every learned transformation is fitted only on that fit call's training rows.

## Model selection and final holdout

Each declared candidate is fitted on training rows and evaluated on the validation partition. With CV, mean training-fold score determines ranking; the independent validation results are still shown separately. Optional tuning supplies just two predefined parameter settings per non-baseline model. No final-test metric is calculated for losing candidates.

The winning **already fitted training-only pipeline** is chosen before calling final-test evaluation. It is not refitted on test or validation rows. `final_test_evaluations=1` is recorded with exact indices. A unit test records evaluation access, modifies only final-test labels and proves that candidate selection scores and winning algorithm stay unchanged. That verifies the program boundary; it cannot prevent a person from manually tuning later experiments after seeing the same holdout.

Scores use the configured primary metric. Regression losses are negated only internally for maximizing selection scores. Scores reported under validation/final-test labels are actual metrics on those separate sets. A negative R² is a legitimate result. ROC-AUC/average precision remain null if their mathematical class-support requirements are missing. Calibration bins summarize model confidence versus observed correctness; they do not calibrate a model or supply guarantees.

## Explanations and errors

Permutation importance uses a bounded validation subset (up to 300 rows), three repeats and the selected primary scorer. It measures change in model score when an original input column is shuffled. Negative values and correlated-feature ambiguity are expected. Tree impurity importance, where available, is additionally recorded over actual encoded columns with its high-cardinality bias noted. Neither method establishes causation.

Classification stores bounded misclassification examples, with false-positive/false-negative labels for binary targets (the second sorted class is positive). Regression stores bounded residual and largest-error examples. Candidate-level error analysis uses validation data; the selected model's final error analysis uses the holdout. Examples contain row index and target/prediction, not copies of raw input feature rows.

## Reproduction

Repeat an experiment from its details page or call the shared service using its saved `ExperimentConfig`. The dataset fingerprint and exact software versions identify the input/environment. Training-only random processes receive the recorded seed; native thread pools and estimator job counts are limited to one. Elapsed times/RSS snapshots may vary, while model outputs are reproducible within the pinned environment. Runtime models/datasets are not distributed in Git: run the deterministic smoke command to reconstruct them. Reproducing after dataset deletion requires uploading the original normalized-equivalent contents again; dataset IDs differ, so select the new dataset record while preserving the other settings.

The supplied datasets are generated, not observations from a real deployment. Sample performance measures only those synthetic tasks. They establish neither real-world business value nor model accuracy on a user's uploaded dataset.
