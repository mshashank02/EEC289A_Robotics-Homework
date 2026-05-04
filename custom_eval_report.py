#!/usr/bin/env python3
"""Summarize the custom rollout bundle with richer directional analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from custom_benchmark_specs import CUSTOM_EPISODE_LABELS
from public_eval import clean_json_value, find_key, normalize_rollout, save_json, to_float


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "course_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", type=Path, required=True, help="Path to the custom rollout .npz file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to the output JSON file.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def custom_episode_label(episode_idx: int, bundle: dict[str, np.ndarray]) -> str:
    episode_label = find_key(bundle, ["episode_label"], required=False)
    if episode_label is not None and len(episode_label) > 0:
        unique = np.unique(episode_label[find_key(bundle, ["episode_id", "episode_ids"], required=False) == episode_idx])
        if unique.size > 0:
            value = unique[0]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return CUSTOM_EPISODE_LABELS[episode_idx % len(CUSTOM_EPISODE_LABELS)]


def classify_direction(label: str) -> str:
    if "combined" in label:
        return "combined"
    if "turn" in label:
        return "forward_turn"
    if "forward" in label:
        return "forward"
    if "lateral" in label:
        return "lateral"
    if "yaw" in label:
        return "yaw"
    return "other"


def classify_magnitude(label: str) -> str:
    if label.endswith("_low"):
        return "low"
    if label.endswith("_high"):
        return "high"
    return "medium"


def compute_per_episode_summary(bundle: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    episode_id = find_key(bundle, ["episode_id", "episode_ids"], required=False)
    if episode_id is None:
        episode_id = np.zeros(len(next(iter(bundle.values()))), dtype=np.int32)

    command_lin = find_key(bundle, ["command_lin_vel_xy", "command_xy", "cmd_lin_vel_xy"])
    measured_lin = find_key(bundle, ["measured_lin_vel_xy", "base_lin_vel_xy", "obs_lin_vel_xy"])
    command_yaw = find_key(bundle, ["command_yaw_rate", "cmd_yaw_rate", "command_ang_vel_z"])
    measured_yaw = find_key(bundle, ["measured_yaw_rate", "base_yaw_rate", "obs_yaw_rate"])
    fell = find_key(bundle, ["fell", "fall", "fall_flag", "terminated_by_fall"], required=False)
    if fell is None:
        fell = np.zeros(command_yaw.shape[0], dtype=bool)
    fell = np.asarray(fell, dtype=bool)

    torques = find_key(bundle, ["joint_torques", "torques", "tau"], required=False)
    joint_vel = find_key(bundle, ["joint_velocities", "joint_vel", "qvel_joints"], required=False)
    foot_slip = find_key(bundle, ["foot_slip_speed", "foot_slip", "foot_slip_proxy"], required=False)

    summaries: list[dict[str, Any]] = []
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        label = custom_episode_label(int(eid), bundle)
        vel_err = np.linalg.norm(command_lin[mask] - measured_lin[mask], axis=-1)
        vx_err = np.abs(command_lin[mask, 0] - measured_lin[mask, 0])
        vy_err = np.abs(command_lin[mask, 1] - measured_lin[mask, 1])
        yaw_err = np.abs(command_yaw[mask] - measured_yaw[mask])
        energy = np.abs(torques[mask] * joint_vel[mask]).sum(axis=-1) if torques is not None and joint_vel is not None else np.array([])
        slip = np.asarray(foot_slip[mask], dtype=np.float32) if foot_slip is not None else np.array([])

        summaries.append(
            {
                "episode_id": int(eid),
                "episode_label": label,
                "direction_group": classify_direction(label),
                "magnitude_group": classify_magnitude(label),
                "num_steps": int(np.sum(mask)),
                "fell": bool(np.any(fell[mask])),
                "velocity_tracking_error": _safe_mean(vel_err),
                "vx_tracking_error": _safe_mean(vx_err),
                "vy_tracking_error": _safe_mean(vy_err),
                "yaw_tracking_error": _safe_mean(yaw_err),
                "energy_proxy": _safe_mean(energy),
                "foot_slip_proxy": _safe_mean(slip),
                "mean_command_vx": _safe_mean(command_lin[mask, 0]),
                "mean_command_vy": _safe_mean(command_lin[mask, 1]),
                "mean_command_yaw": _safe_mean(command_yaw[mask]),
                "mean_measured_vx": _safe_mean(measured_lin[mask, 0]),
                "mean_measured_vy": _safe_mean(measured_lin[mask, 1]),
                "mean_measured_yaw": _safe_mean(measured_yaw[mask]),
            }
        )
    return summaries


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)

    results: list[dict[str, Any]] = []
    for group_name, group_rows in sorted(grouped.items()):
        results.append(
            {
                key: group_name,
                "num_episodes": len(group_rows),
                "fall_rate": float(np.mean([float(row["fell"]) for row in group_rows])),
                "velocity_tracking_error": float(np.mean([row["velocity_tracking_error"] for row in group_rows])),
                "vx_tracking_error": float(np.mean([row["vx_tracking_error"] for row in group_rows])),
                "vy_tracking_error": float(np.mean([row["vy_tracking_error"] for row in group_rows])),
                "yaw_tracking_error": float(np.mean([row["yaw_tracking_error"] for row in group_rows])),
                "energy_proxy": float(np.mean([row["energy_proxy"] for row in group_rows])),
                "foot_slip_proxy": float(np.mean([row["foot_slip_proxy"] for row in group_rows])),
            }
        )
    return results


def compute_overall_metrics(per_episode: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "velocity_tracking_error": float(np.mean([row["velocity_tracking_error"] for row in per_episode])),
        "vx_tracking_error": float(np.mean([row["vx_tracking_error"] for row in per_episode])),
        "vy_tracking_error": float(np.mean([row["vy_tracking_error"] for row in per_episode])),
        "yaw_tracking_error": float(np.mean([row["yaw_tracking_error"] for row in per_episode])),
        "fall_rate": float(np.mean([float(row["fell"]) for row in per_episode])),
        "energy_proxy": float(np.mean([row["energy_proxy"] for row in per_episode])),
        "foot_slip_proxy": float(np.mean([row["foot_slip_proxy"] for row in per_episode])),
    }


def maybe_make_plots(output_dir: Path, per_episode: list[dict[str, Any]], by_direction: list[dict[str, Any]], by_magnitude: list[dict[str, Any]]) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plots: dict[str, str] = {}

    direction_labels = [row["direction_group"] for row in by_direction]
    direction_vel = [row["velocity_tracking_error"] for row in by_direction]
    direction_yaw = [row["yaw_tracking_error"] for row in by_direction]
    direction_fall = [row["fall_rate"] for row in by_direction]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].bar(direction_labels, direction_vel, color="#1f77b4")
    axes[0].set_title("Velocity Tracking Error by Direction Group")
    axes[0].set_ylabel("Planar velocity error")

    axes[1].bar(direction_labels, direction_yaw, color="#ff7f0e")
    axes[1].set_title("Yaw Tracking Error by Direction Group")
    axes[1].set_ylabel("Yaw-rate error")

    axes[2].bar(direction_labels, direction_fall, color="#2ca02c")
    axes[2].set_title("Fall Rate by Direction Group")
    axes[2].set_ylabel("Fall rate")

    path = output_dir / "custom_eval_by_direction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["by_direction"] = str(path)

    magnitude_labels = [row["magnitude_group"] for row in by_magnitude]
    magnitude_vel = [row["velocity_tracking_error"] for row in by_magnitude]
    magnitude_yaw = [row["yaw_tracking_error"] for row in by_magnitude]
    magnitude_slip = [row["foot_slip_proxy"] for row in by_magnitude]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(magnitude_labels, magnitude_vel, marker="o", color="#1f77b4")
    axes[0].set_title("Velocity Tracking Error by Magnitude Group")
    axes[0].set_ylabel("Planar velocity error")

    axes[1].plot(magnitude_labels, magnitude_yaw, marker="o", color="#ff7f0e")
    axes[1].set_title("Yaw Tracking Error by Magnitude Group")
    axes[1].set_ylabel("Yaw-rate error")

    axes[2].plot(magnitude_labels, magnitude_slip, marker="o", color="#d62728")
    axes[2].set_title("Foot Slip Proxy by Magnitude Group")
    axes[2].set_ylabel("Slip proxy")
    axes[2].set_xlabel("Magnitude group")

    path = output_dir / "custom_eval_by_magnitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["by_magnitude"] = str(path)

    episode_labels = [row["episode_label"] for row in per_episode]
    episode_vel = [row["velocity_tracking_error"] for row in per_episode]
    episode_yaw = [row["yaw_tracking_error"] for row in per_episode]
    colors = ["#d62728" if row["fell"] else "#1f77b4" for row in per_episode]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    axes[0].bar(episode_labels, episode_vel, color=colors)
    axes[0].set_title("Per-Episode Velocity Tracking Error")
    axes[0].set_ylabel("Planar velocity error")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(episode_labels, episode_yaw, color=colors)
    axes[1].set_title("Per-Episode Yaw Tracking Error")
    axes[1].set_ylabel("Yaw-rate error")
    axes[1].tick_params(axis="x", rotation=35)

    path = output_dir / "custom_eval_per_episode.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["per_episode"] = str(path)

    return plots


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    bundle = normalize_rollout(dict(np.load(args.rollout_npz, allow_pickle=True)))

    per_episode = compute_per_episode_summary(bundle)
    by_direction = aggregate_by_key(per_episode, "direction_group")
    by_magnitude = aggregate_by_key(per_episode, "magnitude_group")
    overall_metrics = compute_overall_metrics(per_episode)

    output_dir = args.output_json.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = maybe_make_plots(output_dir, per_episode, by_direction, by_magnitude)

    result = {
        "homework_name": config["homework_name"],
        "robot": config["robot"],
        "environment_name": config["environment_name"],
        "num_steps": int(len(next(iter(bundle.values())))),
        "num_episodes": len(per_episode),
        "metrics": overall_metrics,
        "per_episode_summary": per_episode,
        "by_direction_summary": by_direction,
        "by_magnitude_summary": by_magnitude,
        "plot_paths": plot_paths,
    }
    cleaned = clean_json_value(result)
    save_json(args.output_json, cleaned)
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
