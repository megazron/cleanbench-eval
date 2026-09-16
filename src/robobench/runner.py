"""The evaluation loop: an Env protocol you plug your sim/replay into, a Policy
protocol your controller implements, and `evaluate` that runs a suite and returns
per-task results. Two envs and two policies ship so the whole thing runs with no
hardware and no simulator.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from typing import Protocol, runtime_checkable

import numpy as np

from .task import TaskSpec, load_suite, _check_predicate, predicate_names
from .embodiment import check as embodiment_check
from .perturb import Perturbation


# ------------------------------------------------------------------ protocols
@runtime_checkable
class Env(Protocol):
    """A world the policy acts in. Reset to a task, step with an action, and expose
    the SIGNALS the task's success predicate is written over."""

    def reset(self, task: TaskSpec, rng: np.random.Generator) -> dict: ...
    def step(self, action) -> tuple[dict, bool]: ...      # (obs, done)
    def signals(self) -> dict: ...                        # for the predicate


@runtime_checkable
class Policy(Protocol):
    name: str
    embodiments: list          # embodiment tags it can drive
    trained_on: list           # task hashes/ids/suite hashes it trained on

    def reset(self, task: TaskSpec) -> None: ...
    def act(self, obs: dict): ...


# ------------------------------------------------------------------ predicate
def _eval_predicate(expr: str, signals: dict) -> bool:
    """Evaluate a whitelisted predicate against recorded signals. Missing signals
    are a task/env contract error, raised loudly, never silently False."""
    tree = _check_predicate(expr)
    need = predicate_names(expr)
    missing = need - set(signals)
    if missing:
        raise KeyError(
            "success predicate needs signal(s) %s that the env did not record; "
            "the env's signals() must cover every name in the predicate"
            % sorted(missing))
    code = compile(tree, "<predicate>", "eval")
    ns = {"abs": abs, "min": min, "max": max, "__builtins__": {}}
    ns.update(signals)
    return bool(eval(code, ns))          # noqa: S307 - tree is whitelisted above


# ------------------------------------------------------------------ shipped envs
class MockEnv:
    """A scripted toy pick task with no dependencies.

    Progress needs SUSTAINED grip-and-lift: the object rises only on a step where
    the gripper is closed AND lifting, and it slips back down on any step where the
    grip is released. So a policy that flails succeeds essentially never, a policy
    that commits succeeds every time, and a dropped-frame perturbation -- on which a
    good policy holds and therefore lets the object slip -- lowers the score in a way
    you can measure rather than assert."""

    def __init__(self, lift_per_step: float = 0.02, drop_per_step: float = 0.05):
        self.lift_per_step = lift_per_step
        self.drop_per_step = drop_per_step
        self._reset_state()

    def _reset_state(self):
        self.object_z = 0.0
        self.gripper_closed = False
        self._steps = 0
        self._max = 100

    def reset(self, task: TaskSpec, rng: np.random.Generator) -> dict:
        self._reset_state()
        self._max = task.max_steps
        return self._obs()

    def _obs(self) -> dict:
        return {
            "object_px": [320.0, 240.0],
            "object_visible": True,
            "brightness": 1.0,
            "scene_labels": ["cube"],
            "object_z": self.object_z,
            "gripper_closed": self.gripper_closed,
        }

    def step(self, action) -> tuple[dict, bool]:
        self._steps += 1
        grip = float(action[0]) if len(action) > 0 else 0.0
        lift = float(action[1]) if len(action) > 1 else 0.0
        self.gripper_closed = grip > 0.5
        if self.gripper_closed and lift > 0.5:
            self.object_z += self.lift_per_step
        else:
            # not held-and-lifting: the object slips back toward the table
            self.object_z = max(0.0, self.object_z - self.drop_per_step)
        done = self._steps >= self._max
        return self._obs(), done

    def signals(self) -> dict:
        return {"object_z": self.object_z,
                "gripper_closed": self.gripper_closed,
                "steps": self._steps}


class ReplayEnv:
    """Replay recorded episodes from disk. Each episode is a JSON list of frames;
    the final frame's signals decide success. Actions are ignored (the recording is
    fixed), which is exactly what you want when scoring a logged run against a new
    predicate or a new success threshold."""

    def __init__(self, episodes: dict[str, list[dict]]):
        # episodes keyed by task id -> list of frame dicts, each with "signals"
        self.episodes = episodes
        self._frames: list[dict] = []
        self._i = 0

    def reset(self, task: TaskSpec, rng: np.random.Generator) -> dict:
        if task.id not in self.episodes:
            raise KeyError("no recorded episode for task %r" % task.id)
        self._frames = self.episodes[task.id]
        self._i = 0
        return self._frames[0].get("obs", {}) if self._frames else {}

    def step(self, action) -> tuple[dict, bool]:
        self._i += 1
        done = self._i >= len(self._frames) - 1
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        return frame.get("obs", {}), done

    def signals(self) -> dict:
        return self._frames[min(self._i, len(self._frames) - 1)].get("signals", {})


# ------------------------------------------------------------------ shipped policies
@dataclass
class ScriptedPolicy:
    """Drives MockEnv to success: grip then lift, every step it has a fresh frame.
    On a stale (dropped) frame it holds, so heavy frame loss makes it run out of
    steps -- the mechanism that lets a perturbation move the score."""

    name: str = "scripted"
    embodiments: list = field(default_factory=lambda: ["toy_2dof_grip"])
    trained_on: list = field(default_factory=list)

    def reset(self, task: TaskSpec) -> None:
        pass

    def act(self, obs: dict):
        if obs.get("_stale"):
            return [0.0, 0.0]          # no fresh info, hold
        return [1.0, 1.0]              # grip and lift


@dataclass
class RandomPolicy:
    name: str = "random"
    embodiments: list = field(default_factory=lambda: ["toy_2dof_grip"])
    trained_on: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def reset(self, task: TaskSpec) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: dict):
        return [float(self._rng.random()), float(self._rng.random())]


# ------------------------------------------------------------------ results
@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    steps: int
    embodiment_ok: bool
    perturbations: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(policy: Policy, task: TaskSpec, env: Env,
                perturbations: list[Perturbation] | None = None,
                rng: np.random.Generator | None = None) -> EpisodeResult:
    rng = rng or np.random.default_rng(0)
    perturbations = perturbations or []
    comp = embodiment_check(policy.embodiments, task.embodiment.tag)
    policy.reset(task)
    obs = env.reset(task, rng)
    done = False
    steps = 0
    while not done and steps < task.max_steps:
        po = obs
        for p in perturbations:
            po = p.apply(po, rng)
        action = policy.act(po)
        obs, done = env.step(action)
        steps += 1
    success = _eval_predicate(task.success, env.signals())
    return EpisodeResult(
        task_id=task.id, success=success, steps=steps,
        embodiment_ok=comp.ok,
        perturbations=[p.declare() for p in perturbations],
        note=comp.note)


@dataclass
class TaskResult:
    task_id: str
    split: str
    embodiment: str
    embodiment_ok: bool
    trials: int
    successes: int
    steps_to_first_success: int | None
    perturbations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(policy: Policy, suite, env: Env, trials: int = 10, seed: int = 0,
             perturbations: list[Perturbation] | None = None) -> list[TaskResult]:
    """Run every task `trials` times and return per-task results. Deterministic in
    `seed`: trial k of task t uses a derived seed, so a run reproduces exactly."""
    tasks = load_suite(suite) if isinstance(suite, str) else list(suite)
    out: list[TaskResult] = []
    for ti, task in enumerate(tasks):
        succ = 0
        first = None
        for k in range(trials):
            rng = np.random.default_rng((seed * 1_000_003 + ti * 9973 + k) & 0xFFFFFFFF)
            r = run_episode(policy, task, env, perturbations, rng)
            if r.success:
                succ += 1
                if first is None:
                    first = k + 1
        out.append(TaskResult(
            task_id=task.id, split=task.split, embodiment=task.embodiment.tag,
            embodiment_ok=embodiment_check(policy.embodiments, task.embodiment.tag).ok,
            trials=trials, successes=succ, steps_to_first_success=first,
            perturbations=[p.declare() for p in (perturbations or [])]))
    return out
