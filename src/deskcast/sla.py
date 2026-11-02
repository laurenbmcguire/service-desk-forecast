"""SLA breach risk model built only on information available when a ticket is created."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .calendar_utils import business_hour_flags, holidays_between
from .config import CATEGORIES, CHANNELS, GROUPS, PRIORITIES, SLA_TARGET_HOURS

CATEGORICAL = {
    "category": CATEGORIES,
    "priority": PRIORITIES,
    "channel": CHANNELS,
    "assignee_group": GROUPS,
}
NUMERIC = [
    "hour",
    "dow",
    "is_business_hours",
    "is_holiday",
    "queue_open_group",
    "queue_open_total",
    "arrivals_prev_24h_group",
]
QUEUE_FEATURES = ["queue_open_group", "queue_open_total", "arrivals_prev_24h_group"]
FEATURES = list(CATEGORICAL) + NUMERIC


def queue_length_at_creation(created: pd.Series, resolved: pd.Series) -> np.ndarray:
    """Open tickets at each creation time: (# created before t) - (# resolved before t).

    Tickets that are still open (NaT) never leave the queue.
    """
    c = created.to_numpy(dtype="datetime64[ns]")
    r = resolved.to_numpy(dtype="datetime64[ns]")
    order = np.argsort(c, kind="stable")
    c_sorted = c[order]
    r_sorted = np.sort(np.where(np.isnat(r), np.datetime64("2262-01-01"), r))
    created_before = np.searchsorted(c_sorted, c, side="left")
    resolved_before = np.searchsorted(r_sorted, c, side="left")
    return (created_before - resolved_before).astype(int)


def arrivals_in_window(created: pd.Series, hours: float = 24.0) -> np.ndarray:
    """Tickets created in the `hours` before each ticket (excluding itself)."""
    c = np.sort(created.to_numpy(dtype="datetime64[ns]"))
    t = created.to_numpy(dtype="datetime64[ns]")
    window = np.timedelta64(int(hours * 3600), "s")
    return (np.searchsorted(c, t, side="left") - np.searchsorted(c, t - window, side="left")).astype(int)


def label_breaches(tickets: pd.DataFrame, extract_end) -> pd.DataFrame:
    """Resolution hours, SLA target and breach label (NaN when still open and not yet late)."""
    extract_end = pd.Timestamp(extract_end)
    target = tickets["priority"].map(SLA_TARGET_HOURS).astype(float)
    res_h = (tickets["resolved_at"] - tickets["created_at"]) / pd.Timedelta(hours=1)
    age_h = (extract_end - tickets["created_at"]) / pd.Timedelta(hours=1)
    breached = pd.Series(np.nan, index=tickets.index)
    resolved = tickets["resolved_at"].notna()
    breached[resolved] = (res_h[resolved] > target[resolved]).astype(float)
    open_late = ~resolved & (age_h > target)
    breached[open_late] = 1.0
    return pd.DataFrame({"resolution_hours": res_h, "sla_target_hours": target, "breached": breached})


def build_features(tickets: pd.DataFrame, extract_end) -> pd.DataFrame:
    """Creation-time features plus the breach label."""
    df = tickets.copy()
    created = df["created_at"]
    hol = holidays_between(created.min(), created.max())
    hol_idx = pd.to_datetime(list(hol))
    df["hour"] = created.dt.hour
    df["dow"] = created.dt.dayofweek
    df["is_holiday"] = created.dt.normalize().isin(hol_idx).astype(int)
    df["is_business_hours"] = business_hour_flags(pd.DatetimeIndex(created), hol).astype(int)
    df["queue_open_total"] = queue_length_at_creation(created, df["resolved_at"])
    df["queue_open_group"] = 0
    df["arrivals_prev_24h_group"] = 0
    for _, idx in df.groupby("assignee_group").groups.items():
        sub = df.loc[idx]
        df.loc[idx, "queue_open_group"] = queue_length_at_creation(sub["created_at"], sub["resolved_at"])
        df.loc[idx, "arrivals_prev_24h_group"] = arrivals_in_window(sub["created_at"], 24.0)
    labels = label_breaches(df, extract_end)
    return pd.concat([df, labels], axis=1)


def encode(features: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode categoricals with fixed vocabularies (stable across runs)."""
    X = pd.DataFrame(index=features.index)
    for col, vocab in CATEGORICAL.items():
        X[col] = features[col].map({v: i for i, v in enumerate(vocab)}).astype(int)
    for col in NUMERIC:
        X[col] = features[col].astype(float)
    return X


def precision_at_top(y_true, scores, frac: float = 0.10) -> float:
    """Breach rate among the `frac` highest-scored tickets."""
    y = np.asarray(y_true, float)
    s = np.asarray(scores, float)
    k = max(1, int(round(len(s) * frac)))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def _make_model(columns: list[str]) -> HistGradientBoostingClassifier:
    cat_mask = [c in CATEGORICAL for c in columns]
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        early_stopping=False,
        random_state=7,
    )


def train_sla_model(
    feats: pd.DataFrame, test_start, warmup_days: int = 14, importance_sample: int = 8000, seed: int = 0
) -> dict:
    """Time-based split, gradient boosting fit, ROC AUC, precision@top10% and permutation importance."""
    test_start = pd.Timestamp(test_start)
    labelled = feats[feats["breached"].notna()]
    first = labelled["created_at"].min() + pd.Timedelta(days=warmup_days)
    train = labelled[(labelled["created_at"] >= first) & (labelled["created_at"] < test_start)]
    test = labelled[labelled["created_at"] >= test_start]
    X_train, X_test = encode(train), encode(test)
    y_train, y_test = train["breached"].astype(int), test["breached"].astype(int)

    model = _make_model(FEATURES).fit(X_train[FEATURES], y_train)
    p_test = model.predict_proba(X_test[FEATURES])[:, 1]
    auc = roc_auc_score(y_test, p_test)
    p_at_10 = precision_at_top(y_test, p_test, 0.10)
    base = float(y_test.mean())

    no_queue = [f for f in FEATURES if f not in QUEUE_FEATURES]
    ablation = _make_model(no_queue).fit(X_train[no_queue], y_train)
    auc_no_queue = roc_auc_score(y_test, ablation.predict_proba(X_test[no_queue])[:, 1])

    train_rate = train.groupby("priority")["breached"].mean()
    auc_priority_rule = roc_auc_score(y_test, test["priority"].map(train_rate).to_numpy()) if len(train_rate) > 1 else 0.5

    rng = np.random.default_rng(seed)
    n_imp = min(importance_sample, len(X_test))
    pick = np.sort(rng.choice(len(X_test), size=n_imp, replace=False))
    imp = permutation_importance(
        model, X_test[FEATURES].iloc[pick], y_test.iloc[pick], scoring="roc_auc", n_repeats=5, random_state=seed
    )
    importances = (
        pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values(["importance", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    by_priority = test.assign(p=p_test).groupby("priority").agg(
        tickets=("breached", "size"), breach_rate=("breached", "mean"), mean_risk=("p", "mean")
    )
    return {
        "model": model,
        "roc_auc": float(auc),
        "roc_auc_without_queue": float(auc_no_queue),
        "roc_auc_priority_rule": float(auc_priority_rule),
        "precision_at_top10": p_at_10,
        "base_rate": base,
        "lift_at_top10": p_at_10 / base if base > 0 else float("nan"),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": train["created_at"].min().date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test["created_at"].max().date().isoformat(),
        "importances": importances,
        "by_priority": by_priority,
        "test_scores": p_test,
        "test_labels": y_test.to_numpy(),
    }
