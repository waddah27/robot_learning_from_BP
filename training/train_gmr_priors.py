"""Train phase-conditioned material priors from registered cutting trials.

The original notebook truncated every trial to the global minimum number of
samples and treated sample index as task phase. Different pauses and local
backtracking therefore placed different physical progress states at the same
phase. This implementation registers each trial by monotone spatial progress
and fits one shared, standardized, categorical material-conditioned model.
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
    # Two categorical contrasts avoid treating material identity as the
    # ordered continuous variable 1, 2, 3 used by the legacy notebook.
    "crck": ("corck_parallel_iter*.csv", "predicted_pose_twist_wrench_crck.npy", (0.0, 0.0)),
    "peno": ("peno_parallel_iter*.csv", "predicted_pose_twist_wrench_peno.npy", (1.0, 0.0)),
    "pvc": ("pcv_parallel_iter*.csv", "predicted_pose_twist_wrench_pvc.npy", (0.0, 1.0)),
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


def _conditional_mean(
    model: GaussianMixture, conditioned: np.ndarray, n_inputs: int
) -> np.ndarray:
    means_x = model.means_[:, :n_inputs]
    means_y = model.means_[:, n_inputs:]
    delta = conditioned[None, :] - means_x
    conditional_means = []
    log_responsibility = []
    for component in range(model.n_components):
        covariance_xx = model.covariances_[component, :n_inputs, :n_inputs]
        covariance_yx = model.covariances_[component, n_inputs:, :n_inputs]
        inverse_xx = np.linalg.inv(covariance_xx)
        conditional_means.append(
            means_y[component] + covariance_yx @ inverse_xx @ delta[component]
        )
        sign, logdet = np.linalg.slogdet(covariance_xx)
        if sign <= 0:
            raise ValueError("non-positive conditional covariance")
        mahalanobis = delta[component] @ inverse_xx @ delta[component]
        log_responsibility.append(
            np.log(model.weights_[component] + 1e-300)
            - 0.5 * (n_inputs * np.log(2.0 * np.pi) + logdet + mahalanobis)
        )
    conditional_means = np.asarray(conditional_means)
    log_responsibility = np.asarray(log_responsibility)
    responsibility = np.exp(log_responsibility - np.max(log_responsibility))
    responsibility /= responsibility.sum()
    return responsibility @ conditional_means


def _fit_regression(
    training: np.ndarray,
    queries: np.ndarray,
    n_inputs: int,
    components: int,
) -> np.ndarray:
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
    for query in queries:
        conditioned = (
            query - scaler.mean_[:n_inputs]
        ) / scaler.scale_[:n_inputs]
        mean_standardized = _conditional_mean(model, conditioned, n_inputs)
        result.append(
            mean_standardized * scaler.scale_[n_inputs:]
            + scaler.mean_[n_inputs:]
        )
    return np.asarray(result)


def fit_shared_model(
    trials_by_material: dict[str, np.ndarray],
    phase: np.ndarray,
    components: int,
) -> dict[str, np.ndarray]:
    """Fit one material-conditioned structured GMM/GMR to all trials."""
    geometry_rows = []
    behavior_rows = []
    for material, trials in trials_by_material.items():
        contrast = np.asarray(MATERIALS[material][2], dtype=float)
        n_trials = len(trials)
        phase_column = np.tile(phase, n_trials)
        contrast_columns = np.tile(contrast, (n_trials * len(phase), 1))
        inputs = np.column_stack([
            phase_column,
            contrast_columns,
            phase_column[:, None] * contrast_columns,
        ])
        flattened = trials.reshape(-1, trials.shape[-1])
        geometry_rows.append(np.column_stack([inputs, flattened[:, :3]]))
        behavior_rows.append(np.column_stack([inputs, flattened[:, 3:]]))

    geometry_training = np.vstack(geometry_rows)
    behavior_training = np.vstack(behavior_rows)
    outputs = {}
    for material in trials_by_material:
        contrast = np.asarray(MATERIALS[material][2], dtype=float)
        contrast_columns = np.tile(contrast, (len(phase), 1))
        queries = np.column_stack([
            phase,
            contrast_columns,
            phase[:, None] * contrast_columns,
        ])
        # The shared model is structured: an affine one-component positional
        # block and a nonlinear multi-component interaction block, both
        # conditioned on the same phase and categorical material variables.
        geometry = _fit_regression(
            geometry_training, queries, n_inputs=5, components=1
        )
        behavior = _fit_regression(
            behavior_training, queries, n_inputs=5, components=components
        )
        outputs[material] = np.column_stack([geometry, behavior])
    return outputs


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
    trials_by_material = {}
    files_by_material = {}
    for material, (pattern, _, _) in MATERIALS.items():
        files = sorted(DATA_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no demonstrations match {pattern}")
        files_by_material[material] = files
        trials_by_material[material] = np.stack([
            load_and_register(path, phase) for path in files
        ])

    priors = fit_shared_model(trials_by_material, phase, args.components)
    for material, (_, filename, _) in MATERIALS.items():
        prior = priors[material]
        rms, maximum, ratio = straightness(prior[:, :3])
        output = args.output_dir / filename
        np.save(output, prior)
        print(
            f"{material}: {len(files_by_material[material])} trials -> {output}; "
            f"orthogonal RMS={1e3*rms:.6f} mm, "
            f"max={1e3*maximum:.6f} mm, length/chord={ratio:.9f}"
        )


if __name__ == "__main__":
    main()
