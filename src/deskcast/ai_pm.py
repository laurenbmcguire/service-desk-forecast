"""AI Project Manager: turns the roadmap, risk register and latest pipeline results into a status report.

The deterministic engine always runs and is what the tests target. If ANTHROPIC_API_KEY
is set, an optional LLM call writes the executive summary from the computed facts only;
any error falls back to a deterministic template so the run never fails because of it.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from pathlib import Path

import yaml

from .config import FEATURE_LABELS

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
HIGH_RISK_SCORE = 15
CRITICAL_RISK_SCORE = 20
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------- loading
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


# ---------------------------------------------------------------- engine
def heat_label(score: int) -> str:
    if score >= HIGH_RISK_SCORE:
        return "High"
    if score >= 8:
        return "Medium"
    return "Low"


def risk_register(risks: list[dict]) -> list[dict]:
    """Risks with score = probability x impact, sorted by score (desc) then id."""
    out = []
    for r in risks:
        score = int(r["probability"]) * int(r["impact"])
        out.append({**r, "score": score, "heat": heat_label(score)})
    return sorted(out, key=lambda r: (-r["score"], str(r["id"])))


def milestone_progress(roadmap: dict, as_of: date) -> list[dict]:
    """% done per milestone from its linked backlog items; overdue if past due and not done."""
    backlog = roadmap.get("backlog", [])
    rows = []
    for m in roadmap.get("milestones", []):
        items = [b for b in backlog if b.get("milestone") == m["id"]]
        done = sum(1 for b in items if b.get("status") == "done")
        if m["status"] == "done":
            pct = 100
        else:
            pct = round(100 * done / len(items)) if items else 0
        due = _as_date(m["due"])
        overdue = m["status"] != "done" and due < as_of
        rows.append({
            "id": m["id"],
            "name": m["name"],
            "due": due.isoformat(),
            "status": m["status"],
            "items_done": done,
            "items_total": len(items),
            "pct_done": pct,
            "overdue": overdue,
            "days_overdue": (as_of - due).days if overdue else 0,
            "done_criteria": m.get("done_criteria", ""),
        })
    return rows


def next_actions(backlog: list[dict], n: int = 3) -> list[dict]:
    """Unblocked todo items ranked by priority, then by number of satisfied dependencies, then id."""
    status = {b["id"]: b.get("status") for b in backlog}
    ready = []
    for b in backlog:
        if b.get("status") != "todo":
            continue
        deps = b.get("depends_on") or []
        if all(status.get(d) == "done" for d in deps):
            ready.append({**b, "deps_satisfied": len(deps)})
    ready.sort(key=lambda b: (PRIORITY_RANK.get(b.get("priority"), 9), -b["deps_satisfied"], str(b["id"])))
    return ready[:n]


def health(milestones: list[dict], backlog: list[dict], risks: list[dict]) -> dict:
    """RAG status from explicit rules.

    RED   : 2+ overdue milestones, or an overdue milestone while a P0 item is open, or any risk scored >= 20.
    AMBER : any overdue milestone, any open P0 item, or any risk scored >= 15.
    GREEN : none of the above.
    """
    overdue = [m for m in milestones if m["overdue"]]
    open_p0 = [b for b in backlog if b.get("priority") == "P0" and b.get("status") != "done"]
    high = [r for r in risks if r["score"] >= HIGH_RISK_SCORE]
    critical = [r for r in risks if r["score"] >= CRITICAL_RISK_SCORE]

    if len(overdue) >= 2 or (overdue and open_p0) or critical:
        rag = "RED"
    elif overdue or open_p0 or high:
        rag = "AMBER"
    else:
        rag = "GREEN"

    parts = []
    if overdue:
        parts.append(f"{len(overdue)} overdue milestone(s) ({', '.join(m['id'] for m in overdue)})")
    if open_p0:
        parts.append(f"{len(open_p0)} open P0 item(s) ({', '.join(b['id'] for b in open_p0)})")
    if high:
        scored = ", ".join("{}={}".format(r["id"], r["score"]) for r in high)
        parts.append(f"{len(high)} risk(s) scored >= {HIGH_RISK_SCORE} ({scored})")
    reason = "; ".join(parts) if parts else "no overdue milestones, no open P0 items, no risks scored >= 15"
    return {
        "rag": rag,
        "reason": reason,
        "overdue_milestones": len(overdue),
        "open_p0": len(open_p0),
        "high_risks": len(high),
    }


def data_bullets(summary: dict) -> list[str]:
    """3-5 plain-language findings taken from reports/summary.json."""
    if not summary:
        return ["No pipeline results yet: run `python -m deskcast run` to produce reports/summary.json."]
    out = []
    bt = summary.get("forecast", {}).get("backtest")
    if bt:
        out.append(
            f"Forecast: the calendar-feature ridge model beats seasonal naive by {bt['mape_improvement_pct']:.0f}% on MAPE "
            f"({bt['model_mape']:.1f}% vs {bt['baseline_mape']:.1f}%) and {bt['mae_improvement_pct']:.0f}% on MAE "
            f"across {bt['folds']} rolling 28-day folds."
        )
    nxt = summary.get("forecast", {}).get("next_28_days")
    if nxt:
        out.append(
            f"Demand: {nxt['total']:,.0f} tickets expected {nxt['start']} to {nxt['end']} "
            f"(80% band {nxt['total_lo']:,.0f} to {nxt['total_hi']:,.0f}); busiest day {nxt['peak_date']} "
            f"at about {nxt['peak_forecast']:.0f}."
        )
    sla = summary.get("sla")
    if sla:
        top = sla["top_features"][0]["feature"] if sla.get("top_features") else "n/a"
        top = FEATURE_LABELS.get(top, top).lower()
        out.append(
            f"SLA risk: ROC AUC {sla['roc_auc']:.3f}; the top 10% riskiest tickets breach {sla['precision_at_top10pct']:.0%} "
            f"of the time vs a {sla['test_breach_rate']:.0%} base rate ({sla['lift_at_top10pct']:.1f}x). Top driver: {top}."
        )
    st = summary.get("staffing")
    if st:
        out.append(
            f"Staffing: peak of {st['peak_agents']} agents ({st['peak_slot']}) and {st['weekly_agent_hours']} agent-hours "
            f"in a typical week for {st['service_level_target']:.0%}/{st['answer_time_seconds']}s, about "
            f"{st['scheduled_fte']} FTE after {st['shrinkage']:.0%} shrinkage."
        )
    d = summary.get("data")
    if d:
        out.append(
            f"Data: {d['tickets']:,} synthetic tickets over {d['days']} days, {d['sla_breach_rate']:.1%} SLA breach rate, "
            f"{len(d.get('incidents', []))} injected incident surges, "
            f"{d['quality_checks']['resolved_before_created']} timestamp-order violations."
        )
    return out[:5]


def change_log_entry(as_of: date, hlth: dict, milestones: list[dict], actions: list[dict], summary: dict) -> str:
    done = sum(1 for m in milestones if m["status"] == "done")
    focus = ", ".join(a["id"] for a in actions) or "none"
    text = f"{as_of.isoformat()}: health {hlth['rag']}; {done}/{len(milestones)} milestones done; next focus {focus}."
    bt = summary.get("forecast", {}).get("backtest") if summary else None
    if bt:
        text += f" Forecast MAPE {bt['model_mape']:.1f}% vs baseline {bt['baseline_mape']:.1f}%."
    sla = summary.get("sla") if summary else None
    if sla:
        text += f" SLA AUC {sla['roc_auc']:.3f}."
    return text


def build_facts(roadmap: dict, risks_doc: dict, summary: dict, as_of: date) -> dict:
    milestones = milestone_progress(roadmap, as_of)
    backlog = roadmap.get("backlog", [])
    risks = risk_register(risks_doc.get("risks", []))
    actions = next_actions(backlog)
    hlth = health(milestones, backlog, risks)
    return {
        "project": roadmap.get("project", "Service Desk Forecast"),
        "as_of": as_of.isoformat(),
        "health": hlth,
        "milestones": milestones,
        "next_actions": [
            {k: a.get(k) for k in ("id", "title", "priority", "owner", "milestone", "depends_on", "deps_satisfied")}
            for a in actions
        ],
        "risks": risks,
        "backlog_counts": {
            s: sum(1 for b in backlog if b.get("status") == s) for s in ("todo", "doing", "done")
        },
        "data_says": data_bullets(summary),
        "change_log": change_log_entry(as_of, hlth, milestones, actions, summary),
    }


# ---------------------------------------------------------------- summary text
def template_summary(facts: dict) -> str:
    """Deterministic executive summary (about 90-120 words) built from the computed facts."""
    h = facts["health"]
    ms = facts["milestones"]
    done = sum(1 for m in ms if m["status"] == "done")
    active = [m for m in ms if m["status"] == "in_progress"]
    planned = [m for m in ms if m["status"] == "planned"]
    actions = facts["next_actions"]
    top_risk = facts["risks"][0] if facts["risks"] else None
    s = f"Overall health is {h['rag']}: {h['reason']}. {done} of {len(ms)} milestones are done"
    if active:
        s += f"; {active[0]['id']} ({active[0]['name']}) is in progress with {active[0]['pct_done']}% of items closed"
    if planned:
        s += f"; {len(planned)} are planned, the next due {planned[0]['due']}"
    s += ". "
    for i in (0, 2):
        if len(facts["data_says"]) > i:
            s += facts["data_says"][i] + " "
    if actions:
        ids = [a["id"] for a in actions]
        s += "Next actions: " + (", ".join(ids[:-1]) + " and " + ids[-1] if len(ids) > 1 else ids[0]) + ". "
    if top_risk:
        s += f"Top risk: {top_risk['id']} ({top_risk['risk'].split(':')[0]}), score {top_risk['score']}."
    return s.strip()


def llm_summary(facts: dict, api_key: str, model: str = DEFAULT_MODEL, opener=None, timeout: float = 30.0) -> str | None:
    """Ask the Anthropic Messages API for a grounded executive summary. Returns None on any failure."""
    opener = opener or urllib.request.urlopen
    prompt = (
        "You are the AI project manager for a portfolio analytics project. Write a 90-120 word executive summary "
        "for the weekly status report. Use ONLY the facts in the JSON below; do not invent numbers, dates, people, "
        "clients or results. Plain prose, no headings, no bullet points.\n\nFACTS JSON:\n"
        + json.dumps(facts, indent=2, default=str)
    )
    body = json.dumps(
        {"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        words = len(text.split())
        if not text or words < 60 or words > 160:
            return None
        return text
    except Exception:  # noqa: BLE001 - never fail the run because of the LLM
        return None


# ---------------------------------------------------------------- rendering
def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_status_md(facts: dict) -> str:
    h = facts["health"]
    ms_rows = [
        [m["id"], m["name"], m["due"], m["status"], f"{m['pct_done']}%",
         f"{m['items_done']}/{m['items_total']}", f"overdue {m['days_overdue']}d" if m["overdue"] else "on track"]
        for m in facts["milestones"]
    ]
    act_rows = [
        [i + 1, a["id"], a["title"], a["priority"], a["owner"], a.get("milestone") or "-"]
        for i, a in enumerate(facts["next_actions"])
    ]
    risk_rows = [
        [r["id"], r["risk"], r["probability"], r["impact"], r["score"], r["heat"], r["owner"], r["status"]]
        for r in facts["risks"]
    ]
    bc = facts["backlog_counts"]
    parts = [
        f"# Project status: {facts['project']}",
        "",
        f"_As of {facts['as_of']}. Generated by the AI PM (`python -m deskcast pm`) from project/ROADMAP.yml, "
        f"project/RISKS.yml and reports/summary.json. Executive summary source: {facts['executive_summary_source']}._",
        "",
        f"## Overall health: **{h['rag']}**",
        "",
        f"{h['rag']}: {h['reason']}.",
        "",
        "## Executive summary",
        "",
        facts["executive_summary"],
        "",
        "## Milestones",
        "",
        _md_table(["ID", "Milestone", "Due", "Status", "% done", "Items done", "Schedule"], ms_rows),
        "",
        f"Backlog: {bc['done']} done, {bc['doing']} in progress, {bc['todo']} to do.",
        "",
        "## Next 3 actions",
        "",
        _md_table(["#", "ID", "Action", "Priority", "Owner", "Milestone"], act_rows) if act_rows
        else "No unblocked to-do items.",
        "",
        "## Risk register",
        "",
        _md_table(["ID", "Risk", "P", "I", "Score", "Heat", "Owner", "Status"], risk_rows),
        "",
        "## What the data says",
        "",
        *[f"- {b}" for b in facts["data_says"]],
        "",
        "## Decision / change log",
        "",
        f"- {facts['change_log']}",
        "",
    ]
    return "\n".join(parts)


def run_pm(root: str | Path = ".", as_of: str | date | None = None, use_llm: bool = True, opener=None) -> dict:
    root = Path(root)
    as_of_d = _as_date(as_of) if as_of else date.today()
    roadmap = load_yaml(root / "project" / "ROADMAP.yml")
    risks = load_yaml(root / "project" / "RISKS.yml")
    summary = load_summary(root / "reports" / "summary.json")
    facts = build_facts(roadmap, risks, summary, as_of_d)

    text, source = None, "template"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if use_llm and api_key:
        model = os.environ.get("AI_PM_MODEL", DEFAULT_MODEL)
        text = llm_summary(facts, api_key, model, opener=opener)
        if text:
            source = f"LLM ({model})"
    facts["executive_summary"] = text or template_summary(facts)
    facts["executive_summary_source"] = source

    out = root / "project"
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS.md").write_text(render_status_md(facts), encoding="utf-8")
    (out / "status.json").write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")
    return facts
