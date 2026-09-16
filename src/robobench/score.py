"""The scorecard. Separates the axes the eval literature asks benchmarks to keep
apart, and never reports a bare success mean over ten trials as if it were a number
you can trust."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

from .runner import TaskResult


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Far better than mean +/- se
    at the small trial counts robotics evaluation actually uses: 7/10 is not 0.70,
    it is 0.70 with a 95% interval of about [0.40, 0.89], and reporting the point
    alone is how two policies that are statistically indistinguishable get ranked."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    margin = (z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class Scorecard:
    policy: str
    suite_hash: str
    contamination_clean: bool
    leakage_clean: bool
    n_tasks: int
    trials: int
    perturbations: list = field(default_factory=list)
    overall_success: float = 0.0
    overall_ci: tuple = (0.0, 0.0)
    per_embodiment: dict = field(default_factory=dict)
    per_split: dict = field(default_factory=dict)
    cross_embodiment_tasks: list = field(default_factory=list)
    sample_efficiency: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall_ci"] = list(self.overall_ci)
        return d

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @staticmethod
    def from_json(s: str) -> "Scorecard":
        d = json.loads(s)
        d["overall_ci"] = tuple(d.get("overall_ci", (0.0, 0.0)))
        return Scorecard(**d)

    def table(self) -> str:
        lines = []
        gate = "CLEAN" if (self.contamination_clean and self.leakage_clean) else "NOT CLEAN"
        lines.append("robobench scorecard  policy=%s  [%s]" % (self.policy, gate))
        lines.append("suite %s   %d tasks x %d trials"
                     % (self.suite_hash[:19], self.n_tasks, self.trials))
        if self.perturbations:
            lines.append("perturbations: %s"
                         % ", ".join(p["name"] for p in self.perturbations))
        lo, hi = self.overall_ci
        lines.append("overall success  %.2f  95%% CI [%.2f, %.2f]  (in-distribution only)"
                     % (self.overall_success, lo, hi))
        lines.append("")
        lines.append("  %-16s %-14s %6s %8s  %s"
                     % ("task", "split", "succ", "trials", "95% CI"))
        for t in self.tasks:
            lo, hi = wilson_interval(t["successes"], t["trials"])
            flag = "" if t["embodiment_ok"] else "  [x-embodiment]"
            lines.append("  %-16s %-14s %6d %8d  [%.2f, %.2f]%s"
                         % (t["task_id"], t["split"], t["successes"], t["trials"],
                            lo, hi, flag))
        if self.per_embodiment:
            lines.append("")
            for emb, v in sorted(self.per_embodiment.items()):
                lines.append("  embodiment %-20s %.2f (%d/%d)"
                             % (emb, v["success"], v["successes"], v["trials"]))
        for w in self.warnings:
            lines.append("  ! " + w)
        return "\n".join(lines)


def build_scorecard(policy_name: str, results: list[TaskResult], *,
                    suite_hash: str, contamination_clean: bool,
                    leakage_clean: bool, trials: int,
                    perturbations: list | None = None,
                    contamination_note: str = "",
                    leakage_note: str = "") -> Scorecard:
    """Fold per-task results into one card. In-distribution tasks (the policy declared
    the embodiment) drive the headline number; cross-embodiment tasks are reported
    separately, never folded in."""
    perturbations = perturbations or []
    in_dist = [r for r in results if r.embodiment_ok]
    cross = [r for r in results if not r.embodiment_ok]

    succ = sum(r.successes for r in in_dist)
    tot = sum(r.trials for r in in_dist)
    overall = succ / tot if tot else 0.0
    ci = wilson_interval(succ, tot)

    per_emb: dict = {}
    for r in results:
        e = per_emb.setdefault(r.embodiment, {"successes": 0, "trials": 0})
        e["successes"] += r.successes
        e["trials"] += r.trials
    for e in per_emb.values():
        e["success"] = e["successes"] / e["trials"] if e["trials"] else 0.0

    per_split: dict = {}
    for r in in_dist:
        s = per_split.setdefault(r.split, {"successes": 0, "trials": 0})
        s["successes"] += r.successes
        s["trials"] += r.trials
    for s in per_split.values():
        s["success"] = s["successes"] / s["trials"] if s["trials"] else 0.0

    eff = {r.task_id: r.steps_to_first_success for r in results}

    warnings = []
    if not contamination_clean:
        warnings.append("contamination: " + (contamination_note or "holdout/test contaminated"))
    if not leakage_clean:
        warnings.append("leakage: " + (leakage_note or "a task shape leaks across splits"))
    if cross:
        warnings.append("%d task(s) scored on an undeclared embodiment; see "
                        "cross_embodiment_tasks (not in the headline number)" % len(cross))

    return Scorecard(
        policy=policy_name,
        suite_hash=suite_hash,
        contamination_clean=contamination_clean,
        leakage_clean=leakage_clean,
        n_tasks=len(results),
        trials=trials,
        perturbations=perturbations,
        overall_success=overall,
        overall_ci=ci,
        per_embodiment=per_emb,
        per_split=per_split,
        cross_embodiment_tasks=[r.to_dict() for r in cross],
        sample_efficiency=eff,
        tasks=[r.to_dict() for r in results],
        warnings=warnings,
    )
