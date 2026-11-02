import numpy as np
import pandas as pd

from deskcast.sla import FEATURES, build_features, label_breaches, precision_at_top, queue_length_at_creation, train_sla_model


def test_queue_length_counts_open_tickets():
    t = pd.to_datetime
    created = pd.Series(t(["2025-01-01 09:00", "2025-01-01 09:10", "2025-01-01 09:20", "2025-01-01 09:30"]))
    resolved = pd.Series(t(["2025-01-01 09:15", None, "2025-01-01 09:25", "2025-01-01 10:00"]))
    # at 09:00 nothing open; 09:10 -> #1 open; 09:20 -> #2 open (#1 closed 09:15); 09:30 -> #2 open (#3 closed)
    assert list(queue_length_at_creation(created, resolved)) == [0, 1, 1, 1]


def test_breach_labels_handle_censoring():
    t = pd.to_datetime
    df = pd.DataFrame({
        "created_at": t(["2025-01-01 00:00", "2025-01-01 00:00", "2025-01-09 20:00", "2025-01-05 00:00"]),
        "resolved_at": t(["2025-01-01 03:00", "2025-01-01 05:00", None, None]),
        "priority": ["P1", "P1", "P1", "P4"],
    })
    lab = label_breaches(df, "2025-01-10")
    assert lab["breached"].iloc[0] == 0.0  # 3h < 4h
    assert lab["breached"].iloc[1] == 1.0  # 5h > 4h
    assert np.isnan(lab["breached"].iloc[2])  # open 4h, target 4h not yet passed -> unknown
    assert np.isnan(lab["breached"].iloc[3])  # P4 open 5 days, target 120h -> not yet late


def test_precision_at_top():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.1, 0.0, 0.2, 0.05])
    assert precision_at_top(y, s, 0.2) == 0.5
    assert precision_at_top(y, s, 0.3) == 2 / 3


def test_sla_model_uses_creation_time_features_and_learns(small_tickets):
    assert not {"resolved_at", "reopened", "resolution_hours", "handle_minutes", "breached"} & set(FEATURES)
    feats = build_features(small_tickets, "2025-10-01")
    res = train_sla_model(feats, "2025-08-01", importance_sample=2000)
    assert res["roc_auc"] > 0.70
    assert res["precision_at_top10"] > 1.5 * res["base_rate"]
    assert set(res["importances"]["feature"]) == set(FEATURES)
