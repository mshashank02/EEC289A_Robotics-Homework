#!/usr/bin/env python3
"""Deterministic command schedules for custom locomotion analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark_specs import command_for_step


CUSTOM_EPISODE_LABELS = (
    "forward_low",
    "forward_high",
    "lateral_left_low",
    "lateral_left_high",
    "yaw_left_low",
    "yaw_left_high",
    "forward_turn_left",
    "combined_left_low",
    "combined_left_high",
)


def custom_command_script(safe_ranges: dict[str, list[float]], episode_idx: int) -> list[list[float]]:
    """Return a deterministic command schedule for one custom analysis episode."""
    _, vx_max = map(float, safe_ranges["vx"])
    _, vy_max = map(float, safe_ranges["vy"])
    _, yaw_max = map(float, safe_ranges["yaw"])

    scripts = [
        [
            [0.0, 0.0, 0.0],
            [0.25 * vx_max, 0.0, 0.0],
            [0.25 * vx_max, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.75 * vx_max, 0.0, 0.0],
            [0.75 * vx_max, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.30 * vy_max, 0.0],
            [0.0, 0.30 * vy_max, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.75 * vy_max, 0.0],
            [0.0, 0.75 * vy_max, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.30 * yaw_max],
            [0.0, 0.0, 0.30 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75 * yaw_max],
            [0.0, 0.0, 0.75 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.45 * vx_max, 0.0, 0.30 * yaw_max],
            [0.60 * vx_max, 0.0, 0.50 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.30 * vx_max, 0.30 * vy_max, 0.25 * yaw_max],
            [0.40 * vx_max, 0.40 * vy_max, 0.30 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.55 * vx_max, 0.55 * vy_max, 0.40 * yaw_max],
            [0.75 * vx_max, 0.75 * vy_max, 0.55 * yaw_max],
            [0.0, 0.0, 0.0],
        ],
    ]
    return scripts[episode_idx % len(scripts)]


def custom_command_episode_label(episode_idx: int) -> str:
    return CUSTOM_EPISODE_LABELS[episode_idx % len(CUSTOM_EPISODE_LABELS)]


def build_custom_eval_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the full list of custom analysis episodes in readable form."""
    safe_ranges = config["public_eval"]["safe_command_ranges"]
    return [
        {
            "episode_id": idx,
            "episode_label": custom_command_episode_label(idx),
            "segments": custom_command_script(safe_ranges, idx),
        }
        for idx in range(len(CUSTOM_EPISODE_LABELS))
    ]


__all__ = [
    "CUSTOM_EPISODE_LABELS",
    "build_custom_eval_manifest",
    "command_for_step",
    "custom_command_episode_label",
    "custom_command_script",
]
