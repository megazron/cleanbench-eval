"""Embodiment descriptors and the cross-embodiment compatibility gate.

The literature flags cross-robot action generalization as an open milestone. The
smallest honest thing a harness can do about it is REFUSE to fold a policy's score
on an embodiment it never declared it could drive, and to state the transform when
a joint-space policy is scored on an end-effector task or vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable


class EmbodimentError(ValueError):
    """Raised when an embodiment descriptor is malformed."""


@dataclass(frozen=True)
class Embodiment:
    """A minimal, comparable description of the body a task or policy assumes.

    `tag` is the identity used for matching (e.g. "gen3_7dof_2f85"). Two embodiments
    are the SAME body when their tags match; the numeric fields are metadata that a
    report shows and that an action-space transform may consult.
    """

    tag: str
    dof: int
    gripper: str = "none"          # e.g. "2f85", "suction", "none"
    base: str = "fixed"           # "fixed", "mobile", "wheeled", "legged"

    def __post_init__(self) -> None:
        if not self.tag or not isinstance(self.tag, str):
            raise EmbodimentError("embodiment tag must be a non-empty string")
        if int(self.dof) <= 0:
            raise EmbodimentError("dof must be positive, got %r" % (self.dof,))

    @staticmethod
    def from_dict(d: dict) -> "Embodiment":
        if not isinstance(d, dict):
            raise EmbodimentError("embodiment must be a mapping, got %r" % type(d))
        try:
            return Embodiment(
                tag=str(d["tag"]),
                dof=int(d["dof"]),
                gripper=str(d.get("gripper", "none")),
                base=str(d.get("base", "fixed")),
            )
        except KeyError as e:
            raise EmbodimentError("embodiment missing field %s" % e) from None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpace:
    """How a policy's raw action maps onto the joints/pose a task is scored in.

    kind is "joint" or "ee" (end-effector). `dim` is the raw output width. A
    normalizer records the declared transform so a report can state, in one line,
    that a 7-vector joint command was scored on an ee task through `ik`, rather than
    silently comparing incomparable numbers.
    """

    kind: str
    dim: int
    transform: str = "identity"    # "identity", "ik", "fk", "retarget:<tag>"

    def __post_init__(self) -> None:
        if self.kind not in ("joint", "ee"):
            raise EmbodimentError("action kind must be 'joint' or 'ee', got %r" % self.kind)
        if int(self.dim) <= 0:
            raise EmbodimentError("action dim must be positive")

    def describe_against(self, task_kind: str) -> str:
        """One-line statement of what is being compared, for the scorecard."""
        if task_kind == self.kind:
            return "%s->%s direct (%s)" % (self.kind, task_kind, self.transform)
        return "%s->%s via %s" % (self.kind, task_kind, self.transform)


def compatible(policy_embodiments: Iterable[str], task_embodiment: str) -> bool:
    """True iff the policy declared it can drive the task's embodiment."""
    return task_embodiment in set(policy_embodiments)


@dataclass
class CompatReport:
    """The outcome of matching one policy against one task's embodiment."""

    ok: bool
    task_embodiment: str
    policy_embodiments: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def check(policy_embodiments: Iterable[str], task_embodiment: str) -> CompatReport:
    pe = list(policy_embodiments)
    ok = task_embodiment in set(pe)
    note = "" if ok else (
        "policy does not declare embodiment %r; its score on this task is a "
        "cross-embodiment transfer claim and is recorded separately, never folded "
        "into the in-distribution rate" % task_embodiment)
    return CompatReport(ok=ok, task_embodiment=task_embodiment, policy_embodiments=pe, note=note)
