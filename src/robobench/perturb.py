"""Observation perturbations, so robustness is measured and declared, not asserted.

Each perturbation is a named, seeded transform on an observation dict. A result
records exactly which perturbations were applied, because "robust" with no stated
perturbation is a claim about nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Perturbation:
    name: str
    params: dict
    fn: Callable[[dict, np.random.Generator], dict]

    def apply(self, obs: dict, rng: np.random.Generator) -> dict:
        return self.fn(dict(obs), rng)

    def declare(self) -> dict:
        return {"name": self.name, "params": self.params}


def camera_jitter(sigma_px: float = 2.0) -> Perturbation:
    """Add Gaussian noise to any 2D pixel coordinates in the observation."""
    def fn(obs, rng):
        for k, v in list(obs.items()):
            if k.endswith("_px") and isinstance(v, (list, tuple)) and len(v) == 2:
                obs[k] = [float(v[0] + rng.normal(0, sigma_px)),
                          float(v[1] + rng.normal(0, sigma_px))]
        return obs
    return Perturbation("camera_jitter", {"sigma_px": sigma_px}, fn)


def lighting_shift(scale: float = 0.7) -> Perturbation:
    """Scale any intensity/brightness-like scalar, standing in for a lighting change."""
    def fn(obs, rng):
        for k, v in list(obs.items()):
            if k in ("brightness", "intensity", "exposure") and isinstance(v, (int, float)):
                obs[k] = float(v) * scale
        return obs
    return Perturbation("lighting_shift", {"scale": scale}, fn)


def dropped_frames(p: float = 0.2) -> Perturbation:
    """With probability p, mark the observation stale (a dropped frame the policy
    must cope with). The runner treats a stale frame as 'no new information'."""
    def fn(obs, rng):
        if rng.random() < p:
            obs["_stale"] = True
        return obs
    return Perturbation("dropped_frames", {"p": p}, fn)


def scene_distractor(label: str = "unexpected_object") -> Perturbation:
    """Add a distractor label to the scene the task did not declare -- the benign
    cousin of the environmental-jailbreak input the safety literature warns about."""
    def fn(obs, rng):
        d = list(obs.get("scene_labels", []))
        d.append(label)
        obs["scene_labels"] = d
        return obs
    return Perturbation("scene_distractor", {"label": label}, fn)


_REGISTRY = {
    "camera_jitter": camera_jitter,
    "lighting_shift": lighting_shift,
    "dropped_frames": dropped_frames,
    "scene_distractor": scene_distractor,
}


def by_name(name: str, **kw) -> Perturbation:
    if name not in _REGISTRY:
        raise ValueError("unknown perturbation %r; known: %s"
                         % (name, sorted(_REGISTRY)))
    return _REGISTRY[name](**kw)


def available() -> list[str]:
    return sorted(_REGISTRY)
