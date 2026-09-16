"""Contamination control: the part the eval literature actually asks for.

A benchmark score means nothing if the policy trained on the test tasks. Every task
here has a content hash and a split. A policy declares which task hashes (or suite
hashes) it trained or tuned on. The harness then:

  * refuses to call a holdout score "clean" if any holdout task's hash is in that set,
  * reports exactly which tasks are contaminated,
  * and catches near-duplicate tasks that leak the same problem across two splits
    even when their hashes differ (a reworded copy).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

from .task import TaskSpec, load_suite


def manifest(suite: str | list[TaskSpec]) -> dict:
    """A signed inventory of a suite: per-task hash, split, fingerprint, and a suite
    hash over all of them. Written once and shipped WITH results, so a reviewer can
    confirm the suite that was scored is the suite that was published."""
    tasks = load_suite(suite) if isinstance(suite, str) else list(suite)
    entries = []
    for t in tasks:
        entries.append({
            "id": t.id,
            "hash": t.content_hash(),
            "fingerprint": t.fingerprint(),
            "split": t.split,
            "embodiment": t.embodiment.tag,
        })
    entries.sort(key=lambda e: e["id"])
    suite_blob = json.dumps([e["hash"] for e in entries], separators=(",", ":"))
    import hashlib
    suite_hash = "sha256:" + hashlib.sha256(suite_blob.encode()).hexdigest()
    return {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_hash": suite_hash,
        "n_tasks": len(entries),
        "splits": {s: sum(e["split"] == s for e in entries)
                   for s in sorted({e["split"] for e in entries})},
        "tasks": entries,
    }


@dataclass
class ContaminationReport:
    clean: bool
    contaminated_tasks: list[dict] = field(default_factory=list)
    checked_splits: list[str] = field(default_factory=list)
    n_checked: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _declared_set(policy_meta: dict) -> set[str]:
    """Every task-identifying token the policy declared it trained/tuned on: task
    hashes, task ids, and whole suite hashes."""
    s: set[str] = set()
    for key in ("trained_on", "tuned_on"):
        for item in policy_meta.get(key, []) or []:
            s.add(str(item))
    return s


def check_contamination(policy_meta: dict, suite: str | list[TaskSpec],
                        splits=("test", "holdout")) -> ContaminationReport:
    """Flag any task in the checked splits whose hash OR id the policy trained on.

    A whole-suite hash in the declaration contaminates every task in that suite, so
    "I trained on suite X" is caught even without per-task hashes.
    """
    tasks = load_suite(suite) if isinstance(suite, str) else list(suite)
    declared = _declared_set(policy_meta)
    man = manifest(tasks)
    suite_hash = man["suite_hash"]
    suite_declared = suite_hash in declared

    hits = []
    checked = 0
    for t in tasks:
        if t.split not in splits:
            continue
        checked += 1
        h = t.content_hash()
        reason = None
        if suite_declared:
            reason = "policy declares the whole suite hash %s" % suite_hash
        elif h in declared:
            reason = "task hash in policy training set"
        elif t.id in declared:
            reason = "task id %r in policy training set" % t.id
        if reason:
            hits.append({"id": t.id, "split": t.split, "hash": h, "reason": reason})

    clean = not hits
    note = "" if clean else (
        "%d of %d checked task(s) are in the policy's declared training set; a score "
        "on these splits is NOT contamination-clean and must not be reported as such"
        % (len(hits), checked))
    return ContaminationReport(
        clean=clean, contaminated_tasks=hits,
        checked_splits=list(splits), n_checked=checked, note=note)


@dataclass
class LeakageReport:
    clean: bool
    cross_split_duplicates: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def leakage_report(suite: str | list[TaskSpec]) -> LeakageReport:
    """Catch the SAME task appearing in two different splits under different ids.

    Matching is by fingerprint (the task shape without its id), so a holdout task
    copied into dev with a new name and a reworded description is caught even though
    its content hash differs.
    """
    tasks = load_suite(suite) if isinstance(suite, str) else list(suite)
    by_fp: dict[str, list[TaskSpec]] = {}
    for t in tasks:
        by_fp.setdefault(t.fingerprint(), []).append(t)
    dups = []
    for fp, group in by_fp.items():
        splits = {t.split for t in group}
        if len(group) > 1 and len(splits) > 1:
            dups.append({
                "fingerprint": fp,
                "tasks": [{"id": t.id, "split": t.split} for t in group],
                "reason": "same task shape appears in splits %s" % sorted(splits),
            })
    dups.sort(key=lambda d: d["fingerprint"])
    clean = not dups
    note = "" if clean else (
        "%d task shape(s) leak across splits; a holdout that also appears in dev is "
        "not a holdout" % len(dups))
    return LeakageReport(clean=clean, cross_split_duplicates=dups, note=note)
