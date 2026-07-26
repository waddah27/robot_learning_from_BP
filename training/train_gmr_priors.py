"""Train phase-conditioned material priors from registered cutting trials.

The original notebook truncated every trial to the global minimum number of
samples and treated sample index as task phase.  Different pauses and local
backtracking therefore placed different physical progress states at the same
phase.  This implementation registers each trial by monotone spatial progress
along its demonstrated cut before fitting a separate, standardized GMM for
each material.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "CleanData" / "grouped" / "ready"
OUTPUT_DIR = ROOT / "data" / "bp_store"
N_PHASE = 300
FREQUENCY_HZ = 100.0

SIGNALS = [
    "X", "Y", "Z", "R_x", "R_y", "R_z",
    " Fx", " Fy", " Fz", " Tx", " Ty", " Tz",
]
MATERIALS = {
    "crck": ("corck_parallel_iter*.csv", "predicted_pose_twist_wrench_crck.npy"),
    "peno": ("peno_parallel_iter*.csv", "predicted_pose_twist_wrench_peno.npy"),
    "pvc": ("pcv_parallel_iter*.csv", "predicted_pose_twist_wrench_pvc.npy"),
}


def _strict_progress(position: np.ndarray) -> np.ndarray:
    """Return monotone endpoint progress for one straight-cut demonstration."""
    direction = position[-1] - position[0]
    norm_sq = float(direction @ direction)
    if norm_sq < 1e-12:
        raise ValueError("demonstration endpoints are coincident")
    progress = ((position - position[0]) @ direction) / norm_sq
    progress = np.maximum.accumulate(np.clip(progress, 0.0, 1.0))
    progress[0], progress[-1] = 0.0, 1.0
    return progress


def _unique_progress(progress: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last observation at each monotone progress value."""
    keep = np.r_[progress[1:] > progress[:-1] + 1e-10, True]
    return progress[keep], values[keep]


def load_and_register(path: Path, phase: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    values = frame[SIGNALS].to_numpy(dtype=float)
    values[:, 6:12] /= -20.0

    progress = _strict_progress(values[:, :3])
    progress, values = _unique_progress(progress, values)
    registered = np.column_stack([
        np.interp(phase, progress, values[:, column])
        for column in range(values.shape[1])
    ])

    # Derive twist from the registered trajectory.  This is internally
    # consistent with the learned phase coordinate; runtime position velocity
    # is subsequently obtained by the phase chain rule.
    dt = 1.0 / FREQUENCY_HZ
    linear_velocity = np.gradient(registered[:, :3], dt, axis=0)
    angular_velocity = np.gradient(registered[:, 3:6], dt, axis=0)
    return np.column_stack([registered, linear_velocity, angular_velocity])


def _conditional_mean(model: GaussianMixture, value: float) -> np.ndarray:
    delta = value - model.means_[:, 0]
    variance = model.covariances_[:, 0, 0]
    conditional_means = (
        model.means_[:, 1:]
        + model.covariances_[:, 1:, 0] * (delta / variance)[:, None]
    )
    log_responsibility = (
        np.log(model.weights_ + 1e-300)
        - 0.5 * (np.log(2.0 * np.pi * variance) + delta**2 / variance)
    )
    responsibility = np.exp(log_responsibility - np.max(log_responsibility))
    responsibility /= responsibility.sum()
    return responsibility @ conditional_means


def _fit_regression(training: np.ndarray, phase: np.ndarray, components: int) -> np.ndarray:
    scaler = StandardScaler().fit(training)
    standardized = scaler.transform(training)
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=1000,
        n_init=3,
        random_state=42,
    ).fit(standardized)
    result = []
    for value in phase:
        conditioned_value = (value - scaler.mean_[0]) / scaler.scale_[0]
        mean_standardized = _conditional_mean(model, conditioned_value)
        result.append(mean_standardized * scaler.scale_[1:] + scaler.mean_[1:])
    return np.asarray(result)


def fit_material(files: list[Path], phase: np.ndarray, components: int) -> np.ndarray:
    trials = np.stack([load_and_register(path, phase) for path in files])
    phase_column = np.tile(phase, len(trials))
    flattened = trials.reshape(-1, trials.shape[-1])

    # A single Gaussian is the appropriate phase-position model for a straight
    # cut: its conditional mean is affine in phase.  Interaction/orientation
    # channels retain a multi-component mixture to express nonlinear profiles.
    geometry_training = np.column_stack([phase_column, flattened[:, :3]])
    behavior_training = np.column_stack([
        phase_column,
        flattened[:, 3:],
    ])
    geometry = _fit_regression(geometry_training, phase, components=1)
    behavior = _fit_regression(behavior_training, phase, components=components)
    return np.column_stack([geometry, behavior])


def straightness(position: np.ndarray) -> tuple[float, float, float]:
    centered = position - position.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    residual = centered - np.outer(centered @ basis[0], basis[0])
    distances = np.linalg.norm(residual, axis=1)
    chord = np.linalg.norm(position[-1] - position[0])
    length = np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
    return (
        float(np.sqrt(np.mean(distances**2))),
        float(np.max(distances)),
        float(length / max(chord, 1e-12)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    phase = np.linspace(0.0, 1.0, N_PHASE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for material, (pattern, filename) in MATERIALS.items():
        files = sorted(DATA_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no demonstrations match {pattern}")
        prior = fit_material(files, phase, args.components)
        rms, maximum, ratio = straightness(prior[:, :3])
        output = args.output_dir / filename
        np.save(output, prior)
        print(
            f"{material}: {len(files)} trials -> {output}; "
            f"orthogonal RMS={1e3*rms:.6f} mm, "
            f"max={1e3*maximum:.6f} mm, length/chord={ratio:.9f}"
        )


if __name__ == "__main__":
    main()
