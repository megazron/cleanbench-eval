#!/usr/bin/env python3
"""End-to-end example: validate a suite, check a policy's declaration, run the
scripted policy over the mock env, and write a scorecard plus an HTML card.

    python3 examples/run_example.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import yaml

from robobench import (load_suite, manifest, check_contamination, leakage_report,
                       evaluate, MockEnv, ScriptedPolicy)
from robobench.score import build_scorecard
from robobench.report import to_text, write_html
from robobench import perturb

SUITE = os.path.join(HERE, "suite")
META = yaml.safe_load(open(os.path.join(HERE, "policy_meta.yaml")))


def main():
    tasks = load_suite(SUITE)
    print("suite: %d tasks" % len(tasks))

    man = manifest(tasks)
    con = check_contamination(META, tasks)
    lk = leakage_report(tasks)
    print("contamination clean:", con.clean, "--", con.note or "ok")
    print("leakage clean:", lk.clean)

    policy = ScriptedPolicy(embodiments=META["embodiments"])
    policy.trained_on = META["trained_on"]
    policy.name = META["name"]

    # clean run, then a robustness run under dropped frames
    for perts in ([], [perturb.dropped_frames(0.5)]):
        results = evaluate(policy, tasks, MockEnv(), trials=20, seed=7,
                           perturbations=perts)
        card = build_scorecard(
            policy.name, results, suite_hash=man["suite_hash"],
            contamination_clean=con.clean, leakage_clean=lk.clean,
            trials=20, perturbations=[p.declare() for p in perts],
            contamination_note=con.note, leakage_note=lk.note)
        tag = "clean" if not perts else "dropped_frames_0.5"
        print("\n=== %s ===" % tag)
        print(to_text(card))
        out_json = os.path.join(HERE, "scorecard_%s.json" % tag)
        out_html = os.path.join(HERE, "scorecard_%s.html" % tag)
        card.to_json(out_json)
        write_html(card, out_html)
        print("wrote", os.path.basename(out_json), "and", os.path.basename(out_html))


if __name__ == "__main__":
    main()
