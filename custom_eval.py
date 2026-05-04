#!/usr/bin/env python3
"""Custom evaluation suite for richer Go2 locomotion analysis.

This complements the official public benchmark with a larger grid of
constant-command tests. The suite is designed to answer:

1. How well does the policy track commands in different directions?
2. How does performance change with command magnitude?
3. Does the policy stay upright across the evaluation grid?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a PPO checkpoint directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs.")
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to use when building the eval environment.",
    )
    parser.add_argument(
        "--episode-length-steps",
        type=int,
        default=None,
        help="Optional override for the number of control steps per custom-eval episode.",
    )
    parser.add_argument("--render-first-n", type=int, default=0, help="Optional number of first episodes to render.")
    parser.add_argument("--render-width", type=int, default=960, help="Rendered video width.")
    parser.add_argument("--render-height", type=int, default=540, help="Rendered video height.")
    parser.add_argument("--render-camera", type=str, default="track", help="Camera name used for MuJoCo rendering.")
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    return parser.parse_args()


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _safe_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def build_custom_eval_specs(safe_ranges: dict[str, list[float]]) -> list[dict[str, Any]]:
    _, vx_max = map(float, safe_ranges["vx"])
    _, vy_max = map(float, safe_ranges["vy"])
    _, yaw_max = map(float, safe_ranges["yaw"])

    magnitudes = [
        ("low", 0.25),
        ("medium", 0.5),
        ("high", 0.75),
    ]
    direction_templates = [
        ("forward", lambda vx, vy, yaw: [vx, 0.0, 0.0]),
        ("backward", lambda vx, vy, yaw: [-vx, 0.0, 0.0]),
        ("lateral_left", lambda vx, vy, yaw: [0.0, vy, 0.0]),
        ("lateral_right", lambda vx, vy, yaw: [0.0, -vy, 0.0]),
        ("yaw_left", lambda vx, vy, yaw: [0.0, 0.0, yaw]),
        ("yaw_right", lambda vx, vy, yaw: [0.0, 0.0, -yaw]),
        ("forward_left", lambda vx, vy, yaw: [vx, vy, 0.0]),
        ("forward_right", lambda vx, vy, yaw: [vx, -vy, 0.0]),
        ("forward_turn_left", lambda vx, vy, yaw: [vx, 0.0, yaw]),
        ("forward_turn_right", lambda vx, vy, yaw: [vx, 0.0, -yaw]),
        ("combined_left", lambda vx, vy, yaw: [vx, vy, yaw]),
        ("combined_right", lambda vx, vy, yaw: [vx, -vy, -yaw]),
    ]

    specs: list[dict[str, Any]] = []
    episode_id = 0
    for magnitude_label, scale in magnitudes:
        vx = scale * vx_max
        vy = scale * vy_max
        yaw = scale * yaw_max
        for direction_label, command_fn in direction_templates:
            specs.append(
                {
                    "episode_id": episode_id,
                    "direction_label": direction_label,
                    "magnitude_label": magnitude_label,
                    "magnitude_scale": scale,
                    "command": [float(x) for x in command_fn(vx, vy, yaw)],
                }
            )
            episode_id += 1
    return specs


def summarize_episode(spec: dict[str, Any], records: dict[str, list[Any]], episode_length: int) -> dict[str, Any]:
    command_xy = np.asarray(records["command_xy"], dtype=np.float32)
    measured_xy = np.asarray(records["measured_xy"], dtype=np.float32)
    command_yaw = np.asarray(records["command_yaw"], dtype=np.float32)
    measured_yaw = np.asarray(records["measured_yaw"], dtype=np.float32)
    torques = np.asarray(records["joint_torques"], dtype=np.float32)
    joint_velocities = np.asarray(records["joint_velocities"], dtype=np.float32)
    foot_slip = np.asarray(records["foot_slip_speed"], dtype=np.float32)
    fell = np.asarray(records["fell"], dtype=bool)

    velocity_tracking_error = np.linalg.norm(command_xy - measured_xy, axis=-1)
    vx_error = np.abs(command_xy[:, 0] - measured_xy[:, 0])
    vy_error = np.abs(command_xy[:, 1] - measured_xy[:, 1])
    yaw_error = np.abs(command_yaw - measured_yaw)
    energy_proxy = np.abs(torques * joint_velocities).sum(axis=-1)
    base_speed = np.linalg.norm(measured_xy, axis=-1)

    num_steps = int(command_xy.shape[0])
    return {
        "episode_id": int(spec["episode_id"]),
        "direction_label": spec["direction_label"],
        "magnitude_label": spec["magnitude_label"],
        "magnitude_scale": float(spec["magnitude_scale"]),
        "command": [float(x) for x in spec["command"]],
        "num_steps": num_steps,
        "survival_fraction": float(num_steps / max(1, episode_length)),
        "fell": bool(np.any(fell)),
        "velocity_tracking_error": _safe_mean(velocity_tracking_error),
        "vx_tracking_error": _safe_mean(vx_error),
        "vy_tracking_error": _safe_mean(vy_error),
        "yaw_tracking_error": _safe_mean(yaw_error),
        "energy_proxy": _safe_mean(energy_proxy),
        "foot_slip_proxy": _safe_mean(foot_slip),
        "mean_measured_vx": _safe_mean(measured_xy[:, 0]),
        "mean_measured_vy": _safe_mean(measured_xy[:, 1]),
        "mean_measured_yaw_rate": _safe_mean(measured_yaw),
        "mean_measured_speed": _safe_mean(base_speed),
    }


def aggregate_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        group_rows = grouped[key]
        summaries.append(
            {
                group_key: key,
                "num_episodes": len(group_rows),
                "fall_rate": float(np.mean([float(item["fell"]) for item in group_rows])),
                "mean_survival_fraction": float(np.mean([item["survival_fraction"] for item in group_rows])),
                "velocity_tracking_error": float(np.mean([item["velocity_tracking_error"] for item in group_rows])),
                "vx_tracking_error": float(np.mean([item["vx_tracking_error"] for item in group_rows])),
                "vy_tracking_error": float(np.mean([item["vy_tracking_error"] for item in group_rows])),
                "yaw_tracking_error": float(np.mean([item["yaw_tracking_error"] for item in group_rows])),
                "energy_proxy": float(np.mean([item["energy_proxy"] for item in group_rows])),
                "foot_slip_proxy": float(np.mean([item["foot_slip_proxy"] for item in group_rows])),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plots(
    output_dir: Path,
    by_direction: list[dict[str, Any]],
    by_magnitude: list[dict[str, Any]],
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plot_paths: dict[str, str] = {}

    direction_labels = [row["direction_label"] for row in by_direction]
    direction_velocity = [row["velocity_tracking_error"] for row in by_direction]
    direction_yaw = [row["yaw_tracking_error"] for row in by_direction]
    direction_fall = [row["fall_rate"] for row in by_direction]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].bar(direction_labels, direction_velocity, color="#1f77b4")
    axes[0].set_title("Velocity Tracking Error by Direction")
    axes[0].set_ylabel("Mean planar velocity error")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(direction_labels, direction_yaw, color="#ff7f0e")
    axes[1].set_title("Yaw Tracking Error by Direction")
    axes[1].set_ylabel("Mean absolute yaw-rate error")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(direction_labels, direction_fall, color="#2ca02c")
    axes[2].set_title("Fall Rate by Direction")
    axes[2].set_ylabel("Fall rate")
    axes[2].tick_params(axis="x", rotation=45)

    direction_plot = output_dir / "tracking_by_direction.png"
    fig.savefig(direction_plot, dpi=180)
    plt.close(fig)
    plot_paths["tracking_by_direction"] = str(direction_plot)

    magnitude_labels = [row["magnitude_label"] for row in by_magnitude]
    magnitude_velocity = [row["velocity_tracking_error"] for row in by_magnitude]
    magnitude_yaw = [row["yaw_tracking_error"] for row in by_magnitude]
    magnitude_fall = [row["fall_rate"] for row in by_magnitude]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(magnitude_labels, magnitude_velocity, marker="o", color="#1f77b4")
    axes[0].set_title("Velocity Tracking Error by Command Magnitude")
    axes[0].set_ylabel("Mean planar velocity error")

    axes[1].plot(magnitude_labels, magnitude_yaw, marker="o", color="#ff7f0e")
    axes[1].set_title("Yaw Tracking Error by Command Magnitude")
    axes[1].set_ylabel("Mean absolute yaw-rate error")

    axes[2].plot(magnitude_labels, magnitude_fall, marker="o", color="#2ca02c")
    axes[2].set_title("Fall Rate by Command Magnitude")
    axes[2].set_ylabel("Fall rate")
    axes[2].set_xlabel("Magnitude")

    magnitude_plot = output_dir / "tracking_by_magnitude.png"
    fig.savefig(magnitude_plot, dpi=180)
    plt.close(fig)
    plot_paths["tracking_by_magnitude"] = str(magnitude_plot)

    return plot_paths


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]
    media = stack["media"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    episode_length = int(round(config["public_eval"]["episode_length_seconds"] / env_cfg.ctrl_dt))
    if args.episode_length_steps is not None:
        episode_length = int(args.episode_length_steps)
    env_cfg.episode_length = episode_length

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    eval_specs = build_custom_eval_specs(config["public_eval"]["safe_command_ranges"])
    rng = jax.random.PRNGKey(int(config["seed"]) + 314)

    episode_ids = []
    direction_labels = []
    magnitude_labels = []
    command_xy = []
    measured_xy = []
    command_yaw = []
    measured_yaw = []
    fell = []
    joint_torques = []
    joint_velocities = []
    foot_slip_speed = []

    per_episode_rows: list[dict[str, Any]] = []
    rendered_episodes: list[dict[str, str]] = []

    for spec in eval_specs:
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        command = np.asarray(spec["command"], dtype=np.float32)
        state = _force_command(state, command, jax)

        trajectory = [state] if spec["episode_id"] < int(args.render_first_n) else []
        records = {
            "command_xy": [],
            "measured_xy": [],
            "command_yaw": [],
            "measured_yaw": [],
            "joint_torques": [],
            "joint_velocities": [],
            "foot_slip_speed": [],
            "fell": [],
        }

        for _step_idx in range(episode_length):
            state = _force_command(state, command, jax)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, command, jax)

            local_linvel = np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32)
            yaw_rate = _safe_float(env.get_gyro(state.data)[2])
            feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], dtype=np.float32)
            slip_speed = np.linalg.norm(feet_vel[:, :2], axis=-1).astype(np.float32)
            done = bool(np.asarray(state.done))

            episode_ids.append(spec["episode_id"])
            direction_labels.append(spec["direction_label"])
            magnitude_labels.append(spec["magnitude_label"])
            command_xy.append(command[:2])
            measured_xy.append(local_linvel)
            command_yaw.append(command[2])
            measured_yaw.append(yaw_rate)
            joint_torques.append(np.asarray(state.data.actuator_force, dtype=np.float32))
            joint_velocities.append(np.asarray(state.data.qvel[6:], dtype=np.float32))
            foot_slip_speed.append(slip_speed)
            fell.append(done)

            records["command_xy"].append(command[:2])
            records["measured_xy"].append(local_linvel)
            records["command_yaw"].append(command[2])
            records["measured_yaw"].append(yaw_rate)
            records["joint_torques"].append(np.asarray(state.data.actuator_force, dtype=np.float32))
            records["joint_velocities"].append(np.asarray(state.data.qvel[6:], dtype=np.float32))
            records["foot_slip_speed"].append(slip_speed)
            records["fell"].append(done)

            if trajectory:
                trajectory.append(state)

            if done:
                break

        if trajectory:
            video_path = output_dir / f"episode_{spec['episode_id']:02d}_{spec['direction_label']}_{spec['magnitude_label']}.mp4"
            frames = env.render(
                trajectory,
                height=int(args.render_height),
                width=int(args.render_width),
                camera=args.render_camera,
            )
            media.write_video(video_path, frames, fps=int(round(1.0 / env.dt)))
            rendered_episodes.append(
                {
                    "episode_id": spec["episode_id"],
                    "direction_label": spec["direction_label"],
                    "magnitude_label": spec["magnitude_label"],
                    "video_path": str(video_path),
                }
            )

        per_episode_rows.append(summarize_episode(spec, records, episode_length))

    rollout_npz = output_dir / "rollout_custom_eval.npz"
    np.savez(
        rollout_npz,
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        direction_label=np.asarray(direction_labels),
        magnitude_label=np.asarray(magnitude_labels),
        command_lin_vel_xy=np.asarray(command_xy, dtype=np.float32),
        measured_lin_vel_xy=np.asarray(measured_xy, dtype=np.float32),
        command_yaw_rate=np.asarray(command_yaw, dtype=np.float32),
        measured_yaw_rate=np.asarray(measured_yaw, dtype=np.float32),
        fell=np.asarray(fell, dtype=bool),
        joint_torques=np.asarray(joint_torques, dtype=np.float32),
        joint_velocities=np.asarray(joint_velocities, dtype=np.float32),
        foot_slip_speed=np.asarray(foot_slip_speed, dtype=np.float32),
    )

    by_direction = aggregate_rows(per_episode_rows, "direction_label")
    by_magnitude = aggregate_rows(per_episode_rows, "magnitude_label")
    plot_paths = maybe_write_plots(output_dir, by_direction, by_magnitude)

    write_csv(output_dir / "per_episode_summary.csv", per_episode_rows)
    write_csv(output_dir / "by_direction_summary.csv", by_direction)
    write_csv(output_dir / "by_magnitude_summary.csv", by_magnitude)

    result = _clean_json(
        {
            "homework_name": config["homework_name"],
            "robot": config["robot"],
            "environment_name": env_name,
            "stage_name": args.stage_name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "episode_length_steps": episode_length,
            "num_episodes": len(eval_specs),
            "suite_description": {
                "direction_labels": sorted({spec["direction_label"] for spec in eval_specs}),
                "magnitude_labels": sorted({spec["magnitude_label"] for spec in eval_specs}),
                "commands_tested": eval_specs,
            },
            "aggregate": {
                "fall_rate": float(np.mean([float(row["fell"]) for row in per_episode_rows])),
                "mean_survival_fraction": float(np.mean([row["survival_fraction"] for row in per_episode_rows])),
                "velocity_tracking_error": float(np.mean([row["velocity_tracking_error"] for row in per_episode_rows])),
                "yaw_tracking_error": float(np.mean([row["yaw_tracking_error"] for row in per_episode_rows])),
                "energy_proxy": float(np.mean([row["energy_proxy"] for row in per_episode_rows])),
                "foot_slip_proxy": float(np.mean([row["foot_slip_proxy"] for row in per_episode_rows])),
            },
            "per_episode_summary": per_episode_rows,
            "by_direction_summary": by_direction,
            "by_magnitude_summary": by_magnitude,
            "rendered_episodes": rendered_episodes,
            "artifacts": {
                "rollout_npz": str(rollout_npz),
                "plot_paths": plot_paths,
            },
        }
    )
    save_json(output_dir / "custom_eval.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
