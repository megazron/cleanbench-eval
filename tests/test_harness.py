import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from robobench.task import TaskSpec, load_suite, TaskError, predicate_names
from robobench.embodiment import Embodiment, ActionSpace, compatible, check as emb_check
from robobench.contamination import manifest, check_contamination, leakage_report
from robobench.runner import (MockEnv, ScriptedPolicy, RandomPolicy, evaluate,
                              run_episode, _eval_predicate)
from robobench.score import wilson_interval, build_scorecard, Scorecard
from robobench.report import to_html
from robobench import perturb

SUITE = os.path.join(ROOT, "examples", "suite")


def _task(tid="t", split="test", success="object_z > 0.05 and gripper_closed",
          max_steps=40, tag="toy_2dof_grip"):
    return TaskSpec.from_dict({
        "id": tid, "description": "d",
        "embodiment": {"tag": tag, "dof": 2},
        "success": success, "max_steps": max_steps, "split": split})


# ---- task ----
def test_load_example_suite():
    tasks = load_suite(SUITE)
    assert {t.id for t in tasks} == {"lift_cube_dev", "lift_cube_test", "lift_high_holdout"}


def test_unsafe_predicate_refused():
    with pytest.raises(TaskError):
        _task(success="__import__('os').system('x')")


def test_attribute_predicate_refused():
    with pytest.raises(TaskError):
        _task(success="obj.z > 1")


def test_predicate_names_finds_signals():
    assert predicate_names("object_z > 0.05 and gripper_closed") == {"object_z", "gripper_closed"}


def test_bad_split_refused():
    with pytest.raises(TaskError):
        _task(split="train")


def test_content_hash_stable_and_id_independent_fingerprint():
    a = _task("a")
    b = _task("b")
    assert a.content_hash() != b.content_hash()      # id is in the hash
    assert a.fingerprint() == b.fingerprint()        # but not in the fingerprint


def test_hash_changes_with_threshold():
    a = _task(success="object_z > 0.05 and gripper_closed")
    b = _task(success="object_z > 0.12 and gripper_closed")
    assert a.content_hash() != b.content_hash()


def test_hash_ignores_whitespace_reformat():
    a = _task(success="object_z>0.05 and gripper_closed")
    b = _task(success="object_z > 0.05  and  gripper_closed")
    assert a.content_hash() == b.content_hash()


def test_duplicate_id_refused(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            "id: same\ndescription: d\nembodiment: {tag: t, dof: 2}\n"
            "success: \"object_z > 0.0\"\n")
    with pytest.raises(TaskError):
        load_suite(str(tmp_path))


# ---- embodiment ----
def test_compatible_and_refusal():
    assert compatible(["a", "b"], "a")
    r = emb_check(["a"], "b")
    assert not r.ok and "cross-embodiment" in r.note


def test_action_space_describe():
    a = ActionSpace("joint", 7, "ik")
    assert "via ik" in a.describe_against("ee")
    assert "direct" in a.describe_against("joint")


def test_bad_embodiment_refused():
    with pytest.raises(Exception):
        Embodiment(tag="", dof=2)


# ---- contamination ----
def test_manifest_has_per_task_hashes():
    m = manifest(SUITE)
    assert m["n_tasks"] == 3
    assert all(e["hash"].startswith("sha256:") for e in m["tasks"])
    assert m["suite_hash"].startswith("sha256:")


def test_contamination_flags_declared_test_task():
    meta = {"trained_on": ["lift_cube_test"]}
    rep = check_contamination(meta, SUITE)
    assert not rep.clean
    assert any(h["id"] == "lift_cube_test" for h in rep.contaminated_tasks)


def test_contamination_clean_when_only_dev_declared():
    meta = {"trained_on": ["lift_cube_dev"]}
    rep = check_contamination(meta, SUITE)
    assert rep.clean


def test_contamination_by_hash():
    tasks = load_suite(SUITE)
    h = next(t.content_hash() for t in tasks if t.id == "lift_high_holdout")
    rep = check_contamination({"trained_on": [h]}, tasks)
    assert not rep.clean and rep.contaminated_tasks[0]["id"] == "lift_high_holdout"


def test_whole_suite_hash_contaminates_everything():
    m = manifest(SUITE)
    rep = check_contamination({"trained_on": [m["suite_hash"]]}, SUITE)
    assert not rep.clean and len(rep.contaminated_tasks) == rep.n_checked


def test_leakage_detects_cross_split_duplicate(tmp_path):
    base = ("description: d\nembodiment: {tag: t, dof: 2}\n"
            "success: \"object_z > 0.05\"\nmax_steps: 40\n")
    (tmp_path / "a.yaml").write_text("id: a\nsplit: dev\n" + base)
    (tmp_path / "b.yaml").write_text("id: b\nsplit: holdout\n" + base)
    lk = leakage_report(str(tmp_path))
    assert not lk.clean
    assert lk.cross_split_duplicates[0]["tasks"]


def test_leakage_clean_on_example_suite():
    assert leakage_report(SUITE).clean


# ---- runner / predicate ----
def test_predicate_eval_and_missing_signal():
    assert _eval_predicate("object_z > 0.05 and gripper_closed",
                           {"object_z": 0.1, "gripper_closed": True})
    with pytest.raises(KeyError):
        _eval_predicate("object_z > 0.05", {"gripper_closed": True})


def test_scripted_beats_random_on_mock():
    tasks = load_suite(SUITE)
    sp = ScriptedPolicy(embodiments=["toy_2dof_grip"])
    rp = RandomPolicy(embodiments=["toy_2dof_grip"])
    rs = evaluate(sp, tasks, MockEnv(), trials=8, seed=3)
    rr = evaluate(rp, tasks, MockEnv(), trials=8, seed=3)
    assert sum(r.successes for r in rs) > sum(r.successes for r in rr)


def test_scripted_solves_dev_task_deterministically():
    tasks = [t for t in load_suite(SUITE) if t.id == "lift_cube_dev"]
    r = evaluate(ScriptedPolicy(embodiments=["toy_2dof_grip"]), tasks, MockEnv(),
                 trials=5, seed=0)
    assert r[0].successes == 5


def test_perturbation_lowers_score():
    tasks = [t for t in load_suite(SUITE) if t.id == "lift_cube_dev"]
    env = MockEnv()
    sp = ScriptedPolicy(embodiments=["toy_2dof_grip"])
    clean = evaluate(sp, tasks, env, trials=30, seed=5)
    drop = evaluate(sp, tasks, env, trials=30, seed=5,
                    perturbations=[perturb.dropped_frames(0.8)])
    assert sum(r.successes for r in drop) < sum(r.successes for r in clean)


def test_run_records_declared_perturbation():
    task = load_suite(SUITE)[0]
    r = run_episode(ScriptedPolicy(embodiments=["toy_2dof_grip"]), task, MockEnv(),
                    perturbations=[perturb.camera_jitter(1.0)])
    assert r.perturbations[0]["name"] == "camera_jitter"


# ---- score ----
def test_wilson_brackets_and_width():
    lo, hi = wilson_interval(7, 10)
    assert lo < 0.7 < hi and (hi - lo) > 0.3
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert abs(wilson_interval(10, 10)[1] - 1.0) < 1e-9


def test_cross_embodiment_excluded_from_headline():
    tasks = load_suite(SUITE)
    # policy declares a DIFFERENT embodiment -> all tasks are cross-embodiment
    pol = ScriptedPolicy(embodiments=["other_body"])
    results = evaluate(pol, tasks, MockEnv(), trials=4, seed=0)
    m = manifest(tasks)
    card = build_scorecard(pol.name, results, suite_hash=m["suite_hash"],
                           contamination_clean=True, leakage_clean=True, trials=4)
    assert card.overall_success == 0.0          # nothing in-distribution
    assert len(card.cross_embodiment_tasks) == 3
    assert any("undeclared embodiment" in w for w in card.warnings)


def test_scorecard_json_round_trip():
    tasks = load_suite(SUITE)
    pol = ScriptedPolicy(embodiments=["toy_2dof_grip"])
    results = evaluate(pol, tasks, MockEnv(), trials=4, seed=0)
    m = manifest(tasks)
    card = build_scorecard(pol.name, results, suite_hash=m["suite_hash"],
                           contamination_clean=False, leakage_clean=True, trials=4,
                           contamination_note="test note")
    card2 = Scorecard.from_json(card.to_json())
    assert card2.overall_success == card.overall_success
    assert card2.contamination_clean is False
    assert "contamination" in " ".join(card2.warnings)


def test_html_renders():
    tasks = load_suite(SUITE)
    pol = ScriptedPolicy(embodiments=["toy_2dof_grip"])
    results = evaluate(pol, tasks, MockEnv(), trials=4, seed=0)
    m = manifest(tasks)
    card = build_scorecard(pol.name, results, suite_hash=m["suite_hash"],
                           contamination_clean=True, leakage_clean=True, trials=4)
    h = to_html(card)
    assert "<html" in h and "scorecard" in h and card.policy in h


# ---- cli ----
def test_cli_validate_and_run(capsys):
    from robobench.cli import main
    assert main(["validate", SUITE]) == 0
    assert main(["run", "scripted", SUITE, "--trials", "3"]) == 0
    out = capsys.readouterr().out
    assert "scorecard" in out


def test_cli_check_contamination_exit_code():
    from robobench.cli import main
    meta = os.path.join(ROOT, "examples", "policy_meta.yaml")
    assert main(["check-contamination", meta, SUITE]) == 1     # deliberately dirty


def test_cli_manifest_json(tmp_path, capsys):
    from robobench.cli import main
    out = tmp_path / "m.json"
    assert main(["manifest", SUITE, "-o", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["n_tasks"] == 3
