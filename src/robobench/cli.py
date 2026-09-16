"""robobench command line: validate a suite, sign a manifest, check a policy's
contamination declaration, run the built-in mock policies, and render a scorecard.

Custom policies and environments are used through the Python API; the CLI ships the
mock policy + mock env so the whole pipeline runs with nothing installed but this.
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from .task import load_suite, TaskError
from .contamination import manifest, check_contamination, leakage_report
from .runner import MockEnv, ScriptedPolicy, RandomPolicy, evaluate
from .score import build_scorecard, Scorecard
from .report import to_text, write_html
from . import perturb as perturb_mod

_POLICIES = {"scripted": ScriptedPolicy, "random": RandomPolicy}


def _load_policy_meta(path: str | None) -> dict:
    if not path:
        return {"name": None, "embodiments": [], "trained_on": []}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _make_policy(name: str, meta: dict):
    if name not in _POLICIES:
        raise SystemExit("unknown built-in policy %r; choose from %s. Custom "
                         "policies use the Python API." % (name, sorted(_POLICIES)))
    pol = _POLICIES[name]()
    if meta.get("embodiments"):
        pol.embodiments = list(meta["embodiments"])
    pol.trained_on = list(meta.get("trained_on", []))
    if meta.get("name"):
        pol.name = str(meta["name"])
    return pol


def _cmd_validate(a) -> int:
    tasks = load_suite(a.suite)
    print("OK: %d task(s) in %s" % (len(tasks), a.suite))
    for t in tasks:
        print("  %-18s split=%-8s emb=%-16s %s"
              % (t.id, t.split, t.embodiment.tag, t.content_hash()[:19]))
    lk = leakage_report(tasks)
    if not lk.clean:
        print("\nLEAKAGE:", lk.note)
        for d in lk.cross_split_duplicates:
            print("  ", d["reason"], [x["id"] for x in d["tasks"]])
        return 1
    return 0


def _cmd_manifest(a) -> int:
    m = manifest(a.suite)
    s = json.dumps(m, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
        print("wrote", a.out, "(suite_hash %s)" % m["suite_hash"][:19])
    else:
        print(s)
    return 0


def _cmd_check(a) -> int:
    meta = _load_policy_meta(a.policy_meta)
    rep = check_contamination(meta, a.suite)
    lk = leakage_report(a.suite)
    print("contamination:", "CLEAN" if rep.clean else "NOT CLEAN")
    if not rep.clean:
        print(" ", rep.note)
        for h in rep.contaminated_tasks:
            print("   -", h["id"], h["split"], h["reason"])
    print("leakage:", "CLEAN" if lk.clean else "NOT CLEAN")
    for d in lk.cross_split_duplicates:
        print("   -", d["reason"], [x["id"] for x in d["tasks"]])
    return 0 if (rep.clean and lk.clean) else 1


def _build_perturbations(names: list[str]):
    return [perturb_mod.by_name(n) for n in (names or [])]


def _cmd_run(a) -> int:
    meta = _load_policy_meta(a.policy_meta)
    policy = _make_policy(a.policy, meta)
    tasks = load_suite(a.suite)
    perts = _build_perturbations(a.perturb)
    results = evaluate(policy, tasks, MockEnv(), trials=a.trials, seed=a.seed,
                       perturbations=perts)
    con = check_contamination(meta, tasks)
    lk = leakage_report(tasks)
    man = manifest(tasks)
    card = build_scorecard(
        policy.name, results, suite_hash=man["suite_hash"],
        contamination_clean=con.clean, leakage_clean=lk.clean,
        trials=a.trials, perturbations=[p.declare() for p in perts],
        contamination_note=con.note, leakage_note=lk.note)
    print(to_text(card))
    if a.out:
        card.to_json(a.out)
        print("\nwrote", a.out)
    return 0


def _cmd_card(a) -> int:
    with open(a.scorecard) as f:
        card = Scorecard.from_json(f.read())
    out = a.out or (a.scorecard.rsplit(".", 1)[0] + ".html")
    write_html(card, out)
    print("wrote", out)
    return 0


def _cmd_selftest(a) -> int:
    import numpy as np
    from .score import wilson_interval
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print("  %-52s %s" % (name, "OK" if cond else "FAIL"))

    lo, hi = wilson_interval(7, 10)
    check("wilson(7,10) brackets 0.70 and is wide", lo < 0.70 < hi and (hi - lo) > 0.3)
    lo0, hi0 = wilson_interval(10, 10)
    check("wilson(10,10) upper bound is 1.0", abs(hi0 - 1.0) < 1e-9)
    # scripted beats random on the mock env
    from .runner import evaluate as _eval
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    suite = os.path.join(here, "examples", "suite")
    if os.path.isdir(suite):
        sp = ScriptedPolicy(embodiments=["toy_2dof_grip"])
        rp = RandomPolicy(embodiments=["toy_2dof_grip"])
        rs = _eval(sp, suite, MockEnv(), trials=6, seed=1)
        rr = _eval(rp, suite, MockEnv(), trials=6, seed=1)
        s_succ = sum(r.successes for r in rs)
        r_succ = sum(r.successes for r in rr)
        check("scripted policy beats random on the mock suite", s_succ > r_succ)
    print("selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robobench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="load and check a task suite")
    v.add_argument("suite")
    v.set_defaults(fn=_cmd_validate)

    m = sub.add_parser("manifest", help="write a signed suite manifest")
    m.add_argument("suite")
    m.add_argument("-o", "--out")
    m.set_defaults(fn=_cmd_manifest)

    c = sub.add_parser("check-contamination", help="check a policy's declaration")
    c.add_argument("policy_meta")
    c.add_argument("suite")
    c.set_defaults(fn=_cmd_check)

    r = sub.add_parser("run", help="run a built-in policy over the mock env")
    r.add_argument("policy", help="scripted | random")
    r.add_argument("suite")
    r.add_argument("--policy-meta", help="yaml with embodiments/trained_on")
    r.add_argument("--perturb", nargs="*", default=[],
                   help="perturbations: " + " ".join(perturb_mod.available()))
    r.add_argument("--trials", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("-o", "--out")
    r.set_defaults(fn=_cmd_run)

    cd = sub.add_parser("card", help="render a scorecard JSON to HTML")
    cd.add_argument("scorecard")
    cd.add_argument("-o", "--out")
    cd.set_defaults(fn=_cmd_card)

    st = sub.add_parser("selftest", help="internal checks, no hardware")
    st.set_defaults(fn=_cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (TaskError, FileNotFoundError, KeyError) as e:
        print("error:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
