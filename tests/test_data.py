from dataclasses import replace

import pytest

from mlforge.config import Settings
from mlforge.datasets import inspect_csv, sample_csv
from mlforge.errors import ForgeError


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b" \n",
        b"a,b\n",
        b",x\n1,2\n",
        b"a,a\n1,2\n",
        b"1,2\n3,4\n",
        b"a,b\n1\n",
        b'a,b\n"broken,1',
        b"a\ninf\n",
        b"a\n-Infinity\n",
        b"a\n1e200\n",
        b"a\n\xff\n",
        b"a\n\x00\n",
    ],
)
def test_invalid_csv(payload, tmp_path):
    with pytest.raises(ForgeError):
        inspect_csv(payload, Settings(tmp_path))


def test_limits_checked_before_unbounded_read(tmp_path):
    settings = Settings(tmp_path)
    for payload, small in (
        (b"a\n1\n2\n", replace(settings, max_rows=1)),
        (b"a,b\n1,2\n", replace(settings, max_columns=1)),
        (b"a\n" + b"x" * 20 + b"\n", replace(settings, max_cell_chars=5)),
        (b"a\n1\n", replace(settings, max_bytes=3)),
    ):
        with pytest.raises(ForgeError):
            inspect_csv(payload, small)


def test_profile_types_missing_duplicates_and_warnings(tmp_path):
    raw = b"record_id,num,category,flag,date,empty,mixed\n1,1,a,true,2025-01-01,,1\n2,3,b,false,2025-01-02,,2\n3,,a,true,2025-01-03,,oops\n3,,a,true,2025-01-03,,oops\n"
    data = inspect_csv(raw, Settings(tmp_path))
    schema = {s["name"]: s for s in data.schema}
    assert schema["flag"]["type"] == "boolean"
    assert schema["record_id"]["role"] == "identifier"
    assert schema["date"]["role"] == "date-like"
    assert schema["empty"]["role"] == "empty"
    assert schema["num"]["missing"] == 2
    assert data.profile["duplicates"] == 1
    assert data.profile["numeric"]["num"]["mean"] == 2
    assert data.profile["categorical"]["category"]["top_values"]["a"] == 3
    assert len(data.frame) == 4


def test_fingerprint_normalized_content_not_filename(forge):
    a = forge.ingest(b" value ,category\r\n1,a\r\n2,b\r\n", "../../alpha.csv")
    b = forge.ingest(b"value,category\n1,a\n2,b\n", "different.csv")
    c = forge.ingest(b"value,category\n1,a\n3,b\n", "alpha.csv")
    assert a["id"] == b["id"] and a["fingerprint"] != c["fingerprint"]
    assert a["name"] == "alpha.csv"
    assert len(forge.store.list("dataset")) == 2
    assert forge.data(a["id"]).fingerprint == a["fingerprint"]


def test_sample_deterministic_and_labeled(forge):
    assert sample_csv("classification") == sample_csv("classification")
    assert sample_csv("classification", seed=1) != sample_csv("classification", seed=2)
    for task in ("classification", "regression"):
        record = forge.sample(task)
        assert "synthetic" in record["source"]
        assert record["rows"] == 600
        assert forge.data(record["id"]).frame["target"].notna().all()


def test_modified_dataset_refused(forge):
    record = forge.sample("classification")
    (forge.settings.home / "datasets" / f"{record['id']}.csv").write_text("x,y\n1,2\n")
    with pytest.raises(ForgeError, match="integrity"):
        forge.data(record["id"])
