from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.ui.common import choose_dataset, forge, heading, show_error
from mlforge.schemas import CLASS_METRICS, MODELS, REG_METRICS, ExperimentConfig


def builder() -> None:
    heading(
        "03 / Design",
        "Build a reproducible experiment",
        "Choose your problem, review validation warnings, then start a bounded background CPU job.",
    )
    record = choose_dataset()
    if record is None:
        return
    columns = [s["name"] for s in record["schema"]]
    target = st.selectbox(
        "Prediction target",
        columns,
        index=columns.index("target") if "target" in columns else len(columns) - 1,
    )
    default_reg = "regression" in record["source"] or "regression" in record["name"]
    task = st.radio(
        "Problem type", ["classification", "regression"], index=int(default_reg), horizontal=True
    )
    available = [c for c in columns if c != target]
    defaults = [
        s["name"]
        for s in record["schema"]
        if s["name"] != target and s["role"] not in ("identifier", "date-like", "constant", "empty")
    ][:30]
    st.caption(
        "Suggested features omit identifier/date/constant/empty columns; you can explicitly include them. This does not delete data."
    )
    with st.form("experiment_configuration"):
        name = st.text_input("Experiment name", value=f"{task.title()} experiment", max_chars=100)
        features = st.multiselect("Feature columns", available, default=defaults)
        candidates = st.multiselect(
            "Candidate algorithms (baseline always included)",
            MODELS,
            default=["baseline", "linear", "tree", "forest"],
        )
        a, b, c = st.columns(3)
        valid_metrics = sorted(CLASS_METRICS if task == "classification" else REG_METRICS)
        desired = "f1_macro" if task == "classification" else "rmse"
        metric = a.selectbox(
            "Primary selection metric", valid_metrics, index=valid_metrics.index(desired)
        )
        seed = b.number_input("Random seed", 0, 2**31 - 1, 42)
        strategy = c.selectbox("Split strategy", ["random", "group", "time"])
        # Both selectors remain visible to make Streamlit form submission predictable.
        a, b = st.columns(2)
        group = a.selectbox("Group column (used for group split)", [None, *available])
        time_key = b.selectbox("Time column (used for time split)", [None, *available])
        a, b, c = st.columns(3)
        val = a.slider("Validation fraction", 0.1, 0.35, 0.2, 0.05)
        test = b.slider("Final test fraction", 0.1, 0.35, 0.2, 0.05)
        folds = c.selectbox("CV folds on training only", [0, 3, 5])
        with st.expander("Preprocessing and CPU budget"):
            numeric = st.selectbox("Numeric imputation", ["median", "mean"])
            categorical = st.selectbox("Categorical imputation", ["most_frequent", "constant"])
            tune = st.checkbox("Try two bounded parameter settings per non-baseline algorithm")
            gap = st.number_input("Time boundary gap (distinct timestamps)", 0, 100, 0)
            budget = st.slider("Wall-clock training budget (seconds)", 5, 180, 120)
        review = st.form_submit_button("Validate and review", use_container_width=True)
        run = st.form_submit_button(
            "Start reviewed experiment", type="primary", use_container_width=True
        )
    if review or run:
        try:
            config = ExperimentConfig(
                name=name,
                dataset_id=record["id"],
                target=target,
                task=task,
                features=features,
                models=candidates,
                primary_metric=metric,
                seed=seed,
                split=strategy,
                group_column=group if strategy == "group" else None,
                time_column=time_key if strategy == "time" else None,
                validation_fraction=val,
                test_fraction=test,
                cv_folds=folds,
                tune=tune,
                numeric_imputation=numeric,
                categorical_imputation=categorical,
                time_gap=gap,
                time_budget_s=budget,
            )
            signature = config.model_dump_json()
            if review:
                preflight = forge().preflight(config)
                st.session_state.reviewed_signature = signature
                st.session_state.preflight = preflight
            elif signature != st.session_state.get("reviewed_signature"):
                st.error(
                    "Validate this exact configuration first; settings changed or have not been reviewed."
                )
            else:
                exp = forge().start(config)
                st.session_state.current_experiment = exp["id"]
                st.success(
                    f"Experiment started: {exp['id']}. Open Training & comparison to follow it."
                )
        except ValidationError as exc:
            for item in exc.errors():
                st.error(item["msg"])
        except Exception as exc:
            show_error(exc)
    if st.session_state.get("preflight"):
        preflight = st.session_state.preflight
        st.subheader("Configuration review")
        st.json(
            {"split_rows": preflight["rows"], "effective_cv_folds": preflight["effective_cv_folds"]}
        )
        for note in preflight["warnings"]:
            st.warning(note)


def details() -> None:
    heading(
        "04 / Evaluate",
        "Experiments & model comparison",
        "Model selection uses validation or training CV. The final holdout is evaluated once, after selection.",
    )
    f = forge()
    f.recover_jobs()
    experiments = f.store.list("experiment")
    if not experiments:
        st.info("No experiments yet. Load a sample in Datasets, then open Experiment builder.")
        return
    ids = [e["id"] for e in experiments]
    index = (
        ids.index(st.session_state.current_experiment)
        if st.session_state.get("current_experiment") in ids
        else 0
    )
    names = {e["id"]: e for e in experiments}
    selected = st.selectbox(
        "Experiment",
        ids,
        index=index,
        format_func=lambda key: f"{names[key]['name']} · {names[key]['status']} · {key[:6]}",
    )
    exp = f.store.get("experiment", selected)
    st.session_state.current_experiment = selected
    st.write(f"**Status:** {exp['status']} · {exp.get('progress', '')}")
    st.caption(f"Dataset fingerprint: {exp['dataset_fingerprint']}")
    if exp["status"] in ("queued", "running"):
        st.info(
            "Training runs in a separate process, one experiment at a time. Refresh to see progress."
        )
        if st.button("Refresh training status"):
            st.rerun()
        if st.button("Cancel this experiment"):
            f.cancel(selected)
            st.rerun()
    if exp.get("error"):
        st.error(exp["error"])
    candidates = exp.get("candidates", [])
    if candidates:
        metric = exp["config"]["primary_metric"]
        table = [
            {
                "algorithm": c["algorithm"],
                "parameters": json.dumps(c["parameters"]),
                "status": c["status"],
                f"validation_{metric}": c.get("validation", {}).get(metric),
                "selection_score_higher_is_better": c.get("selection_score"),
                "seconds": round(c["duration_s"], 3),
                "error": c.get("error", ""),
            }
            for c in candidates
        ]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        with st.expander("Inspect candidate validation errors, metrics and CV scores"):
            number = st.selectbox(
                "Candidate validation details",
                list(range(len(candidates))),
                format_func=lambda i: f"{candidates[i]['algorithm']} · {candidates[i]['parameters']} · {candidates[i]['status']}",
            )
            st.json(candidates[number])
    if exp["status"] != "succeeded":
        with st.expander("Saved configuration"):
            st.json(exp["config"])
        return
    result = exp["result"]
    choice = result["candidates"][result["selected_index"]]
    metric = result["selection_metric"]
    a, b, c = st.columns(3)
    a.metric(
        f"Selected validation {metric}",
        f"{choice['validation'][metric]:.4f}"
        if choice["validation"].get(metric) is not None
        else "Unavailable",
    )
    b.metric(
        f"Final holdout {metric}",
        f"{result['final_test'][metric]:.4f}"
        if result["final_test"].get(metric) is not None
        else "Unavailable",
    )
    c.metric("Experiment duration", f"{result['duration_s']:.2f} s")
    st.success(
        f"Selected: {result['selected_algorithm']} · selection source: {result['selection_source']}"
    )
    st.caption(result["selection_rationale"])
    evaluation, explanation, config_tab, registry = st.tabs(
        ["Final holdout", "Explain & inspect", "Reproducibility", "Register model"]
    )
    with evaluation:
        st.write(
            f"Train / validation / test rows: {result['training_rows']} / {result['validation_rows']} / {result['test_rows']}"
        )
        if exp["config"]["task"] == "classification":
            report = result["final_test"]
            st.subheader("Confusion matrix · rows actual, columns predicted")
            st.dataframe(
                pd.DataFrame(
                    report["confusion_matrix"], index=report["classes"], columns=report["classes"]
                ),
                use_container_width=True,
            )
            st.dataframe(pd.DataFrame(report["classification_report"]).T, use_container_width=True)
            st.write(
                {
                    k: report[k]
                    for k in (
                        "accuracy",
                        "balanced_accuracy",
                        "roc_auc",
                        "average_precision",
                        "brier_score",
                    )
                }
            )
            st.caption(
                "Accuracy can mask class imbalance. Probabilities are estimator outputs, not guaranteed calibrated confidence."
            )
            if report["calibration"]:
                st.dataframe(pd.DataFrame(report["calibration"]), use_container_width=True)
            st.dataframe(pd.DataFrame(report["errors"]), use_container_width=True)
        else:
            report = result["final_test"]
            st.write({k: report[k] for k in ("mae", "rmse", "r2", "median_absolute_error")})
            st.caption(
                "R² may be negative; that means performance worse than a mean reference on this holdout. No confidence interval is fabricated."
            )
            st.scatter_chart(pd.DataFrame(report["residuals"]), x="prediction", y="residual")
            st.dataframe(pd.DataFrame(report["largest_errors"]), use_container_width=True)
    with explanation:
        importance = result["feature_importance"]
        st.write(importance["method"])
        st.caption(importance["note"])
        if importance["values"]:
            st.bar_chart(pd.DataFrame(importance["values"]).set_index("feature")[["mean"]])
            st.dataframe(
                pd.DataFrame(importance["values"]), hide_index=True, use_container_width=True
            )
        if importance.get("impurity_encoded_features"):
            with st.expander("Tree impurity importance on encoded features"):
                st.caption(importance["impurity_note"])
                st.dataframe(
                    pd.DataFrame(importance["impurity_encoded_features"]),
                    hide_index=True,
                    use_container_width=True,
                )
        for note in result["warnings"]:
            st.warning(note)
        for candidate in result["candidates"]:
            for note in candidate.get("warnings", []):
                st.warning(f"{candidate['algorithm']}: {note}")
    with config_tab:
        st.json(exp["config"])
        st.json(result["environment"])
        st.write(
            f"Artifact size: {exp['artifact']['bytes']:,} bytes · SHA-256: {exp['artifact']['sha256']}"
        )
        st.download_button(
            "Download experiment metadata",
            json.dumps(exp, indent=2),
            file_name=f"experiment-{selected[:8]}.json",
            mime="application/json",
        )
        st.warning(
            "Repeatedly changing settings after inspecting the same holdout contaminates its interpretation. Reserve external untouched data for final claims."
        )
        if st.button("Reproduce saved configuration as a new experiment"):
            try:
                new = f.start(ExperimentConfig.model_validate(exp["config"]))
                st.session_state.current_experiment = new["id"]
                st.rerun()
            except Exception as exc:
                show_error(exc)
    with registry:
        st.write("Registration is an explicit choice. Review validation and holdout results first.")
        name = st.text_input(
            "Model family name", value=f"{exp['config']['task']}-model", max_chars=100
        )
        if st.button("Register selected pipeline", type="primary"):
            try:
                model = f.register(selected, name)
                st.success(
                    f"Registered {model['name']} v{model['version']} · ID {model['id']}. Open Model registry or Playground."
                )
            except Exception as exc:
                show_error(exc)
