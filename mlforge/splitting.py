from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from mlforge.errors import ForgeError
from mlforge.schemas import ExperimentConfig


@dataclass
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    warnings: list[str]
    purged: list[int]


def build_split(frame: pd.DataFrame, y: pd.Series, config: ExperimentConfig) -> Split:
    ids = np.arange(len(frame))
    notes, purged = [], []
    if len(frame) < 30:
        raise ForgeError(
            "Use at least 30 labeled rows for separate training, validation and final holdout sets"
        )
    if config.split == "random":
        try:
            trainval, test = train_test_split(
                ids,
                test_size=config.test_fraction,
                random_state=config.seed,
                stratify=y if config.task == "classification" else None,
            )
            train, validation = train_test_split(
                trainval,
                test_size=config.validation_fraction / (1 - config.test_fraction),
                random_state=config.seed + 1,
                stratify=y.iloc[trainval] if config.task == "classification" else None,
            )
        except ValueError as exc:
            raise ForgeError(
                "Stratified splitting is impossible for these class counts/fractions; add data or change the split configuration"
            ) from exc
    elif config.split == "group":
        groups = frame[config.group_column]
        if groups.isna().any() or groups.nunique() < 6:
            raise ForgeError("Group splitting needs at least six distinct, nonmissing groups")
        trainval, test = next(
            GroupShuffleSplit(
                n_splits=1, test_size=config.test_fraction, random_state=config.seed
            ).split(ids, y, groups)
        )
        a, b = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=config.validation_fraction / (1 - config.test_fraction),
                random_state=config.seed + 1,
            ).split(trainval, y.iloc[trainval], groups.iloc[trainval])
        )
        train, validation = trainval[a], trainval[b]
    else:
        times = pd.to_datetime(frame[config.time_column], errors="coerce", utc=True)
        if times.isna().any() or times.nunique() < 10:
            raise ForgeError("Time splitting needs at least ten valid distinct timestamps")
        unique = np.sort(times.unique())
        ntest = max(1, math.ceil(len(unique) * config.test_fraction))
        nval = max(1, math.ceil(len(unique) * config.validation_fraction))
        cut1, cut2 = len(unique) - ntest - nval, len(unique) - ntest
        if cut1 <= config.time_gap or nval <= config.time_gap:
            raise ForgeError("Time gap leaves too few timestamp groups; reduce the gap or add data")
        train = ids[times.isin(unique[: cut1 - config.time_gap])]
        validation = ids[times.isin(unique[cut1 : cut2 - config.time_gap])]
        test = ids[times.isin(unique[cut2:])]
        purged = sorted(set(ids) - set(train) - set(validation) - set(test))
        train = np.array(sorted(train, key=lambda i: times.iloc[i]))
        validation = np.array(sorted(validation, key=lambda i: times.iloc[i]))
        test = np.array(sorted(test, key=lambda i: times.iloc[i]))
        notes.append(
            f"Time split purged {len(purged)} rows at boundaries; gap is measured in distinct timestamp groups"
        )
    if min(len(train), len(validation), len(test)) < 3:
        raise ForgeError("Each split needs at least three rows")
    if config.task == "classification" and set(y.iloc[train]) != set(y):
        raise ForgeError(
            "Training split lacks a target class; adjust groups/time boundaries or collect more data"
        )
    folds = []
    if config.cv_folds:
        n = config.cv_folds
        if config.split == "random":
            if config.task == "classification" and y.iloc[train].value_counts().min() < n:
                notes.append(
                    "Too few training examples per class for requested CV; using the separate validation holdout instead"
                )
            else:
                splitter = (
                    StratifiedKFold(n, shuffle=True, random_state=config.seed)
                    if config.task == "classification"
                    else KFold(n, shuffle=True, random_state=config.seed)
                )
                folds = [(train[a], train[b]) for a, b in splitter.split(train, y.iloc[train])]
        elif config.split == "group":
            groups = frame[config.group_column].iloc[train]
            if groups.nunique() < n:
                notes.append(
                    "Too few training groups for requested CV; using the separate validation holdout instead"
                )
            else:
                folds = [
                    (train[a], train[b])
                    for a, b in GroupKFold(n).split(train, y.iloc[train], groups)
                ]
        else:
            times = pd.to_datetime(frame[config.time_column], utc=True)
            unique_train = np.sort(times.iloc[train].unique())
            chunks = np.array_split(unique_train, n + 1)
            if any(len(chunk) == 0 for chunk in chunks):
                notes.append(
                    "Too few training timestamps for CV; using separate validation holdout"
                )
            else:
                for i in range(1, len(chunks)):
                    earlier = np.concatenate(chunks[:i])
                    if config.time_gap:
                        earlier = earlier[: -config.time_gap]
                    a = train[times.iloc[train].isin(earlier)]
                    b = train[times.iloc[train].isin(chunks[i])]
                    if len(a) < 3 or len(b) < 3:
                        folds = []
                        notes.append(
                            "Time CV would have an empty/small fold; using separate validation holdout"
                        )
                        break
                    folds.append((a, b))
    if config.task == "classification" and any(set(y.iloc[a]) != set(y) for a, _ in folds):
        folds = []
        notes.append(
            "A training CV fold lacked a class; using the separate validation holdout instead"
        )
    return Split(np.asarray(train), np.asarray(validation), np.asarray(test), folds, notes, purged)
