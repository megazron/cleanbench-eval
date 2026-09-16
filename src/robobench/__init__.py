"""robobench -- reproducible, contamination-controlled evaluation plumbing for robot policies.

This is the harness a shared robotics benchmark needs, not the benchmark itself.
It makes any task suite comparable: every task is content-hashed and split-labelled,
every policy declares the embodiments it supports and the tasks it trained on, and a
score is refused rather than quietly reported when the holdout is contaminated.
"""
from .task import TaskSpec, load_suite, TaskError
from .embodiment import Embodiment, ActionSpace, compatible
from .contamination import manifest, check_contamination, leakage_report
from .runner import Env, Policy, MockEnv, ReplayEnv, ScriptedPolicy, RandomPolicy, evaluate, run_episode
from .score import Scorecard, wilson_interval
from . import perturb

__version__ = "0.1.0"

__all__ = [
    "TaskSpec", "load_suite", "TaskError",
    "Embodiment", "ActionSpace", "compatible",
    "manifest", "check_contamination", "leakage_report",
    "Env", "Policy", "MockEnv", "ReplayEnv", "ScriptedPolicy", "RandomPolicy",
    "evaluate", "run_episode",
    "Scorecard", "wilson_interval", "perturb",
    "__version__",
]
