"""TaskSpec: a comparable, hashable description of one robot task, loaded from YAML.

A task suite is a directory of these. The point of the class is that the SAME task
produces the SAME content hash on any machine, so two papers can prove they ran the
same thing, and a contamination check can prove a policy did not train on it.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict

import yaml

from .embodiment import Embodiment, EmbodimentError


class TaskError(ValueError):
    """Raised when a task spec is malformed or its predicate is unsafe."""


SPLITS = ("dev", "test", "holdout")

# The only nodes a success predicate may contain. A predicate is data, loaded from
# a file, so it is parsed and whitelisted, never eval'd. This is the same discipline
# a config-driven robot stack needs everywhere: a value from a file is not code.
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Call, ast.Attribute,
)
_ALLOWED_FUNCS = {"abs", "min", "max"}


def _check_predicate(expr: str) -> ast.Expression:
    """Parse a success predicate and reject anything but arithmetic/comparison/bool.

    Returns the parsed tree so `runner` can compile it once. A predicate that calls
    anything other than abs/min/max, or reaches an attribute, is refused here rather
    than at run time on the robot.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise TaskError("success predicate is not a valid expression: %s" % e) from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise TaskError(
                "success predicate contains a disallowed construct %s; only "
                "arithmetic, comparisons, and/or/not, and abs/min/max are allowed"
                % type(node).__name__)
        if isinstance(node, ast.Attribute):
            raise TaskError("success predicate may not use attributes (`.`)")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise TaskError(
                    "success predicate may only call %s" % sorted(_ALLOWED_FUNCS))
    return tree


def predicate_names(expr: str) -> set[str]:
    """The free signal names a predicate reads (so the runner knows what to record)."""
    tree = _check_predicate(expr)
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in _ALLOWED_FUNCS}


@dataclass
class ObservationSpec:
    """What the policy is given. Kept lightweight but explicit, because 'a slightly
    different camera setup' is exactly what makes two papers' numbers incomparable."""

    camera: str = "none"
    width: int = 0
    height: int = 0
    intrinsics: list[float] = field(default_factory=list)   # fx, fy, cx, cy
    rate_hz: float = 0.0

    @staticmethod
    def from_dict(d: dict | None) -> "ObservationSpec":
        d = d or {}
        return ObservationSpec(
            camera=str(d.get("camera", "none")),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            intrinsics=[float(x) for x in d.get("intrinsics", [])],
            rate_hz=float(d.get("rate_hz", 0.0)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskSpec:
    id: str
    description: str
    embodiment: Embodiment
    success: str                       # predicate over recorded signals
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    max_steps: int = 100
    tolerance: float = 0.0
    split: str = "test"
    tags: list[str] = field(default_factory=list)
    seed_assets: dict = field(default_factory=dict)   # e.g. {"scene": "sha256:..."}
    _source: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise TaskError("task id is required")
        if self.split not in SPLITS:
            raise TaskError("split must be one of %s, got %r" % (SPLITS, self.split))
        if int(self.max_steps) <= 0:
            raise TaskError("max_steps must be positive")
        _check_predicate(self.success)      # refuse an unsafe predicate at load

    @staticmethod
    def from_dict(d: dict, source: str = "") -> "TaskSpec":
        if not isinstance(d, dict):
            raise TaskError("task must be a mapping")
        try:
            emb = Embodiment.from_dict(d["embodiment"])
        except KeyError:
            raise TaskError("task %r missing 'embodiment'" % d.get("id")) from None
        except EmbodimentError as e:
            raise TaskError("task %r: %s" % (d.get("id"), e)) from None
        for req in ("id", "description", "success"):
            if req not in d:
                raise TaskError("task %r missing %r" % (d.get("id"), req))
        return TaskSpec(
            id=str(d["id"]),
            description=str(d["description"]),
            embodiment=emb,
            success=str(d["success"]),
            observation=ObservationSpec.from_dict(d.get("observation")),
            max_steps=int(d.get("max_steps", 100)),
            tolerance=float(d.get("tolerance", 0.0)),
            split=str(d.get("split", "test")),
            tags=[str(t) for t in d.get("tags", [])],
            seed_assets=dict(d.get("seed_assets", {})),
            _source=source,
        )

    def canonical(self) -> dict:
        """The content that defines the task, in a deterministic form for hashing.

        Deliberately EXCLUDES the source path and the free-text description, and
        sorts everything, so a task moved between directories or reworded in its
        description keeps its identity, while any change to what is actually tested
        changes the hash.
        """
        return {
            "id": self.id,
            "embodiment": self.embodiment.to_dict(),
            "success": _normalize_expr(self.success),
            "observation": self.observation.to_dict(),
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "tags": sorted(self.tags),
            "seed_assets": self.seed_assets,
        }

    def content_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()

    def fingerprint(self) -> str:
        """A looser identity for near-duplicate detection across splits: the shape of
        the task without its id, so the same task relabelled into two splits collides."""
        c = self.canonical()
        c.pop("id")
        blob = json.dumps(c, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embodiment"] = self.embodiment.to_dict()
        d["observation"] = self.observation.to_dict()
        d.pop("_source", None)
        return d


def _normalize_expr(expr: str) -> str:
    """Whitespace-insensitive canonical form of a predicate, so reformatting it does
    not change the task's identity while changing a threshold does."""
    tree = _check_predicate(expr)
    return ast.dump(tree)


def load_task(path: str) -> TaskSpec:
    with open(path) as f:
        d = yaml.safe_load(f)
    return TaskSpec.from_dict(d, source=os.path.abspath(path))


def load_suite(directory: str) -> list[TaskSpec]:
    """Every *.yaml / *.yml task under `directory`, sorted by id. Raises on a
    duplicate id, because two tasks that answer to one name is a silent merge."""
    if not os.path.isdir(directory):
        raise TaskError("no such suite directory: %s" % directory)
    paths = sorted(glob.glob(os.path.join(directory, "*.yaml"))
                   + glob.glob(os.path.join(directory, "*.yml")))
    tasks: list[TaskSpec] = []
    seen: dict[str, str] = {}
    for p in paths:
        t = load_task(p)
        if t.id in seen:
            raise TaskError("duplicate task id %r in %s and %s"
                            % (t.id, seen[t.id], p))
        seen[t.id] = p
        tasks.append(t)
    if not tasks:
        raise TaskError("suite %s contains no tasks" % directory)
    return sorted(tasks, key=lambda t: t.id)
