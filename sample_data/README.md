# Reproducible synthetic samples

The UI's sample buttons and `python -m mlforge.cli sample --task classification` (or `regression`) call `mlforge/datasets.py:sample_csv`. Each default sample has **600 rows**, six numeric signals, a categorical segment, a boolean feature, a likely identifier and a `target`. A fixed seed (42) introduces bounded missingness in one numeric and one categorical input.

Classification uses `sklearn.datasets.make_classification`: three classes, six numeric features, four informative and one redundant signal. Regression uses `make_regression`: six numeric signals, five informative and Gaussian noise (18). Auxiliary segment/boolean features are synthetic and are not asserted to be predictive. The identifier is excluded from default experiment features. These are generated demonstration problems, not real customer, medical, financial or production data.

No third-party observed dataset is redistributed and no license is asserted over someone else's measurements. Generator source and its use are covered by scikit-learn's BSD-3-Clause project license. References: [make_classification](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.make_classification.html), [make_regression](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.make_regression.html), [scikit-learn license](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/COPYING).

Generated CSVs are stored in the ignored runtime directory when loaded. Reconstruct them through UI/CLI rather than committing datasets or trained artifacts. A fresh clone requires no credentials, external data download or GPU to complete either workflow.
