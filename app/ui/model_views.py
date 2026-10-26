from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.ui.common import forge, heading, show_error
from mlforge.schemas import PredictionRequest


def choose_model():
    models = [m for m in forge().store.list("model") if m["status"] == "registered"]
    if not models:
        st.info(
            "No registered models yet. Train an experiment, inspect its results, then register the selected pipeline."
        )
        return None
    by_id = {m["id"]: m for m in models}
    key = st.selectbox(
        "Registered model",
        list(by_id),
        format_func=lambda k: f"{by_id[k]['name']} · v{by_id[k]['version']} · {by_id[k]['task']}",
    )
    return by_id[key]


def registry() -> None:
    heading(
        "05 / Registry",
        "Versioned, traceable models",
        "Every pipeline carries its feature schema, experiment, dataset fingerprint and compatibility metadata.",
    )
    model = choose_model()
    if model is None:
        return
    st.write(f"**Model ID:** `{model['id']}` · **Version:** {model['version']}")
    st.dataframe(pd.DataFrame(model["feature_schema"]), use_container_width=True, hide_index=True)
    st.json(
        {
            k: model[k]
            for k in (
                "experiment_id",
                "dataset_fingerprint",
                "primary_metric",
                "created_at",
                "status",
                "bytes",
                "environment",
                "preprocessing",
            )
        }
    )
    versions = [m for m in forge().store.list("model") if m["name"] == model["name"]]
    st.dataframe(
        pd.DataFrame(
            [
                {k: m[k] for k in ("id", "version", "created_at", "status", "experiment_id")}
                for m in versions
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Delete this local artifact"):
        confirmed = st.checkbox(f"Confirm deletion of {model['name']} v{model['version']}")
        if st.button("Delete model artifact", disabled=not confirmed):
            try:
                forge().delete_model(model["id"])
                st.rerun()
            except Exception as exc:
                show_error(exc)


def example_row(model: dict) -> dict:
    row = {}
    for spec in model["feature_schema"]:
        baseline = model["reference"][spec["name"]]
        if spec["type"] == "number":
            row[spec["name"]] = baseline["mean"]
        elif spec["type"] == "boolean":
            row[spec["name"]] = True
        else:
            row[spec["name"]] = next(iter(baseline["probabilities"]), None)
    return row


def playground() -> None:
    heading(
        "06 / Inference",
        "Make a real prediction",
        "Use the exact registered preprocessing-plus-model pipeline. Enter one or more JSON rows; null means impute, missing fields are rejected.",
    )
    model = choose_model()
    if model is None:
        return
    text = st.text_area(
        "Prediction rows (JSON array, maximum 256)",
        value=json.dumps([example_row(model)], indent=2),
        height=260,
        key=f"prediction-{model['id']}",
    )
    if st.button("Predict with registered model", type="primary"):
        try:
            payload = PredictionRequest(rows=json.loads(text))
            result = forge().predict(model["id"], payload.rows)
            st.success(
                f"{len(result.predictions)} predictions · {result.duration_ms:.2f} ms · model v{result.model_version}"
            )
            st.dataframe(
                pd.DataFrame({"prediction": result.predictions}),
                use_container_width=True,
                hide_index=True,
            )
            if result.probabilities is not None:
                st.dataframe(
                    pd.DataFrame(result.probabilities, columns=result.classes),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Probabilities are model estimates, not calibrated guarantees.")
            st.json(result.model_dump())
        except (ValueError, ValidationError):
            st.error(
                "Provide a valid JSON array of 1–256 objects with the registered feature fields."
            )
        except Exception as exc:
            show_error(exc)
    with st.expander("Use the same model through REST"):
        st.code(
            f"POST http://127.0.0.1:8000/predict/{model['id']}\nContent-Type: application/json\n\n"
            + json.dumps({"rows": [example_row(model)]}, indent=2),
            language="text",
        )
