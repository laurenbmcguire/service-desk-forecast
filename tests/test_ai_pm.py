import io
import json
from datetime import date

import pytest
import yaml

from deskcast import ai_pm

ROADMAP = {
    "project": "Test Project",
    "milestones": [
        {"id": "M1", "name": "Done thing", "due": "2026-01-10", "status": "done"},
        {"id": "M2", "name": "Late thing", "due": "2026-02-01", "status": "in_progress"},
        {"id": "M3", "name": "Future thing", "due": "2026-06-01", "status": "planned"},
    ],
    "backlog": [
        {"id": "B1", "title": "a", "priority": "P1", "status": "done", "owner": "Lauren", "milestone": "M1", "depends_on": []},
        {"id": "B2", "title": "b", "priority": "P2", "status": "done", "owner": "Lauren", "milestone": "M2", "depends_on": []},
        {"id": "B3", "title": "c", "priority": "P1", "status": "todo", "owner": "AI PM", "milestone": "M2", "depends_on": ["B1"]},
        {"id": "B4", "title": "d", "priority": "P1", "status": "todo", "owner": "Lauren", "milestone": "M2", "depends_on": []},
        {"id": "B5", "title": "e", "priority": "P0", "status": "todo", "owner": "Lauren", "milestone": "M3", "depends_on": ["B4"]},
        {"id": "B6", "title": "f", "priority": "P2", "status": "todo", "owner": "Lauren", "milestone": "M3", "depends_on": ["B1", "B2"]},
        {"id": "B7", "title": "g", "priority": "P3", "status": "doing", "owner": "Lauren", "milestone": "M3", "depends_on": []},
    ],
}
RISKS = {"risks": [
    {"id": "R1", "risk": "low", "probability": 1, "impact": 2, "owner": "x", "mitigation": "m", "status": "open"},
    {"id": "R2", "risk": "high", "probability": 4, "impact": 4, "owner": "x", "mitigation": "m", "status": "open"},
    {"id": "R3", "risk": "mid", "probability": 3, "impact": 3, "owner": "x", "mitigation": "m", "status": "open"},
]}
SUMMARY = {
    "forecast": {
        "backtest": {"folds": 6, "model_mape": 15.0, "baseline_mape": 32.1, "mape_improvement_pct": 53.3,
                     "mae_improvement_pct": 23.3, "model_mae": 13.2, "baseline_mae": 17.2},
        "next_28_days": {"start": "2026-07-01", "end": "2026-07-28", "total": 2688, "total_lo": 2130,
                         "total_hi": 3306, "peak_date": "2026-07-06", "peak_forecast": 154.5},
    },
    "sla": {"roc_auc": 0.838, "precision_at_top10pct": 0.536, "test_breach_rate": 0.155, "lift_at_top10pct": 3.46,
            "top_features": [{"feature": "dow", "importance": 0.16}]},
}


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "project" / "ROADMAP.yml").write_text(yaml.safe_dump(ROADMAP))
    (tmp_path / "project" / "RISKS.yml").write_text(yaml.safe_dump(RISKS))
    (tmp_path / "reports" / "summary.json").write_text(json.dumps(SUMMARY))
    return tmp_path


def test_engine_rules():
    as_of = date(2026, 3, 1)
    ms = ai_pm.milestone_progress(ROADMAP, as_of)
    assert [m["pct_done"] for m in ms] == [100, 33, 0]
    assert [m["overdue"] for m in ms] == [False, True, False] and ms[1]["days_overdue"] == 28
    risks = ai_pm.risk_register(RISKS["risks"])
    assert [(r["id"], r["score"], r["heat"]) for r in risks] == [("R2", 16, "High"), ("R3", 9, "Medium"), ("R1", 2, "Low")]
    # B5 (P0) is blocked by B4; P1 ties are broken by satisfied dependencies, then id
    assert [a["id"] for a in ai_pm.next_actions(ROADMAP["backlog"])] == ["B3", "B4", "B6"]
    h = ai_pm.health(ms, ROADMAP["backlog"], risks)
    assert h["rag"] == "RED"  # overdue milestone while a P0 item is open
    assert "M2" in h["reason"] and "B5" in h["reason"] and "R2=16" in h["reason"]
    ms_ok = ai_pm.milestone_progress(ROADMAP, date(2026, 1, 15))
    assert ai_pm.health(ms_ok, ROADMAP["backlog"], risks)["rag"] == "AMBER"
    calm = [r for r in risks if r["score"] < 15]
    backlog_no_p0 = [b for b in ROADMAP["backlog"] if b["priority"] != "P0"]
    assert ai_pm.health(ms_ok, backlog_no_p0, calm)["rag"] == "GREEN"


def test_status_output_is_deterministic_without_llm(repo, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f1 = ai_pm.run_pm(repo, as_of="2026-03-01")
    md1 = (repo / "project" / "STATUS.md").read_text()
    f2 = ai_pm.run_pm(repo, as_of="2026-03-01")
    md2 = (repo / "project" / "STATUS.md").read_text()
    assert md1 == md2 and f1 == f2
    assert f1["executive_summary_source"] == "template"
    words = len(f1["executive_summary"].split())
    assert 60 <= words <= 140
    assert "As of 2026-03-01" in md1 and "## Next 3 actions" in md1 and "| R2 |" in md1
    status = json.loads((repo / "project" / "status.json").read_text())
    assert status["health"]["rag"] == "RED" and len(status["data_says"]) >= 3


def test_llm_success_and_fallback_never_hit_network(repo, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured = {}

    def fake_ok(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        captured["url"] = req.full_url
        text = " ".join(["word"] * 100)
        return io.BytesIO(json.dumps({"content": [{"type": "text", "text": text}]}).encode())

    facts = ai_pm.run_pm(repo, as_of="2026-03-01", opener=fake_ok)
    assert facts["executive_summary_source"].startswith("LLM")
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["X-api-key"] == "test-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"

    def fake_fail(req, timeout=None):
        raise OSError("network down")

    facts = ai_pm.run_pm(repo, as_of="2026-03-01", opener=fake_fail)
    assert facts["executive_summary_source"] == "template"
