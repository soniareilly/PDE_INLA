from dataclasses import dataclass
from typing import Any
import numpy as np
from collections.abc import Callable


@dataclass(frozen=True)
class TimeConfig:
    nt: int
    t_init: float
    t_final: float
    first_observation_time: float
    observation_dt: float

@dataclass(frozen=True)
class ForwardConfig:
    dim: int
    mesh_vertices: int
    mesh_path: str
    target_path: str
    kappa: float
    sigma_true: float
    prior_mean: float
    time: TimeConfig

@dataclass
class AdvDiffSetup:
    mesh: Any
    Vh: Any
    Vh2: Any
    wind_velocity: Any
    true_initial_condition: Any
    targets: np.ndarray

@dataclass
class LowRankObjective:
    objective: Callable[[np.ndarray], float]
    eigenvalues: np.ndarray | None
    eigenvectors: Any
    pretheta: np.ndarray | None
    sketching_matrix: Any