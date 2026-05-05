# Sim-to-Real Quadruped Locomotion Report Draft

## Problem Setup and Baseline Behavior

This project uses the course `Go2JoystickFlatTerrain` task built on MuJoCo Playground, Brax PPO, and MJX. The policy observes a 48D actor state containing local linear velocity, gyro, gravity, joint position error, joint velocity, last action, and the commanded planar motion `[vx, vy, yaw_rate]`. The action is a 12D joint-position offset added to the nominal standing pose and applied through a PD controller. During training, the critic also receives privileged simulator-only signals.

The course curriculum is split into two stages. Stage 1 is a forward-only warm start, while stage 2 expands the command distribution to include backward motion, lateral motion, and yaw. I restored the stage 1 checkpoint before stage 2, kept domain randomization enabled, and trained stage 2 with PPO using 1024 environments and 128 eval environments. The final successful run also used a runtime override that extended stage 2 from the default 5M timesteps to 10M timesteps.

The baseline failure mode was clear: the policy could walk forward and remain upright, but pure lateral commands were mostly ignored. In the early custom evaluation bundle (`artifacts/custom_eval_bundle_all_videos_1/custom_eval.json`), the lateral episodes commanded `vy=0.15` and `vy=0.20`, but the robot only achieved mean measured lateral velocities of about `-0.0009` and `-0.0017`. The same run also struggled on combined commands and even fell once during a forward-turn episode. This is consistent with a policy that learned a safe low-motion strategy for non-forward commands rather than learning a true lateral gait.

The baseline custom metrics were:

| Run | Velocity err | `vy` err | Yaw err | Fall rate | Energy | Foot slip |
|---|---:|---:|---:|---:|---:|---:|
| Early baseline custom eval | 0.1394 | 0.0825 | 0.1916 | 0.1111 | 12.38 | 0.0942 |
| Final custom eval | 0.0460 | 0.0197 | 0.0637 | 0.0000 | 20.72 | 0.1911 |

So the main problem was not stability in general. The main problem was that the baseline did not convert lateral commands into actual sideways motion.

## Design Choices and Reasoning

### 1. Curriculum and Command Distribution

The first major change was to make stage 2 actually look like stage 2. In the earlier code, stage 2 was still too close to a forward-dominant curriculum, which meant the policy could continue exploiting the stage 1 prior. I changed the stage 2 command ranges and keep probabilities in `configs/course_config.json` so that `vy` and yaw stayed active more often, while `vx` became less dominant. I also added a mode-based command sampler in `go2_pg_env/joystick.py` that distinguishes between:

- forward mode
- lateral mode
- yaw mode
- combined mode
- near-standstill exploration

The goal was to make the policy see clean single-axis lateral commands often enough to learn them, while still preserving mixed-command training for the final benchmark.

This change alone helped expose the policy to the right data distribution, but it did not fully solve the behavior problem. The robot was seeing more lateral commands, but it still preferred to respond conservatively.

### 2. Reward Reshaping for Lateral Motion

The more important change was reward design. The early custom logs suggested that the robot had found a loophole: for pure lateral commands, it was cheaper to stay close to the nominal pose and avoid energetic stepping than to learn a true sidestep gait. That pointed to a reward imbalance rather than a command-distribution problem alone.

The final reward changes in `go2_pg_env/joystick.py` were:

- increased `tracking_lin_vel` from `1.0` to `1.5`
- reduced `tracking_sigma` from `0.25` in earlier iterations to `0.08` in the final run, which made tracking errors matter more sharply
- changed linear tracking from a symmetric XY penalty to a lateral-biased penalty:
  `0.35 * vx_error + 3.5 * vy_error`
- reduced several posture and foot penalties that were suppressing sideways stepping:
  - `orientation`: `-5.0 -> -3.5`
  - `pose`: `0.5/0.2` in earlier variants -> `0.12` in the final version
  - `feet_clearance`: `-2.0 -> -1.2`
  - `feet_height`: `-0.2 -> -0.1`
  - `feet_slip`: `-0.1 -> -0.06`
- added command-dependent gating so that when `|vy|` is large, the policy is allowed to tilt, deviate from the nominal pose, and tolerate some slip/clearance cost while executing a lateral maneuver

The key design principle was: if the task demands a lateral gait, the reward should stop punishing the policy for looking less like a straight-walking gait.

### 3. One Failed Idea

One intermediate idea was to reshape the `vx` and `vy` weights dynamically based on the lateral ratio of the command. That seemed reasonable at first, because the reward would emphasize `vy` only when the command was mostly lateral. However, that change by itself was not enough and was later reverted. The policy still had strong incentives from orientation, pose, and foot penalties to remain conservative.

The lesson was that lateral tracking is not just a command-sampling problem and not just a one-line reward-weight problem. It required coordinated changes across both the task reward and the penalties that define what the policy considers "safe."

## Official Benchmark and Additional Evaluation

### Official Benchmark

The public benchmark uses four episodes: forward only, lateral only, yaw only, and combined. Comparing an early benchmark bundle to the final benchmark bundle shows that tracking improved substantially:

| Run | Velocity err | Yaw err | Fall rate | Energy | Foot slip | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Early public eval (`new_1`) | 0.0961 | 0.1028 | 0.0000 | 3.47 | 0.0128 | 0.9986 |
| Final public eval (`new_5`) | 0.0661 | 0.0798 | 0.0000 | 7.05 | 0.0508 | 0.9829 |

The interesting part is that the final policy tracked commands better, but its official composite score was slightly lower. This happened because the final policy was more dynamic: it moved laterally and combined commands more aggressively, which increased energy use and slip. The benchmark’s thresholded normalization makes this easy to miss. In other words, the early policy looked "cheap" and safe, but it was not actually solving the lateral-control problem well.

This is why I relied on additional evaluation rather than the official benchmark alone.

### Additional Custom Evaluation

The custom 9-episode evaluation was much more diagnostic because it broke performance down by direction and magnitude. This exposed the baseline’s main weakness and made it possible to verify the fix.

The strongest evidence is the lateral-only episodes:

| Episode | Mean command `vy` | Mean measured `vy` (early) | Mean measured `vy` (final) |
|---|---:|---:|---:|
| Lateral low | 0.092 to 0.150 range across evals | about `-0.0009` | `0.0823` |
| Lateral high | 0.132 to 0.200 range across evals | about `-0.0017` | `0.1256` |

In the early custom run, the policy effectively did not move sideways. In the final run, it tracked lateral motion closely and consistently without falling.

The combined-command episodes improved as well. In the final custom bundle, the high combined-left episode commanded approximately `[0.336, 0.124, 0.252]` and achieved mean measured values `[0.308, 0.116, 0.241]`, which is close tracking across all three command channels.

The final custom run also improved stability and consistency:

- custom velocity tracking error improved from `0.1394` to `0.0460`
- custom `vy` tracking error improved from `0.0825` to `0.0197`
- custom yaw tracking error improved from `0.1916` to `0.0637`
- fall rate improved from `0.1111` to `0.0`

The stage-2 training summary also improved markedly. The early baseline stage-2 summary had `eval/episode_reward=17.98` at the final checkpoint and average eval episode length around `896` steps, while the final `run_updates_6` stage-2 summary reached `eval/episode_reward=41.85` with average eval episode length `1000.0` and zero termination events in the final evaluation.

## Insights and Observations About Policy Behavior

The main insight from this project is that a policy can appear stable and still fail the task in an important way. The baseline robot was not crashing frequently during pure lateral commands. Instead, it was exploiting the reward by remaining near a low-motion posture, which kept energy and slip small but produced almost no sideways motion.

The second insight is that command coverage matters, but reward balance matters more. Expanding the stage-2 command distribution was necessary because the policy needed to see lateral commands often. However, that alone did not force the policy to execute them. The real breakthrough came from making lateral errors expensive and making lateral maneuvers less over-penalized.

The third insight is that the official benchmark can hide failure modes when its thresholds saturate. In my case, an early policy got a very high composite score even though the custom evaluation clearly showed that lateral-only motion was nearly absent. For that reason, I think the custom evaluation was more useful for development and debugging.

Finally, there was a clear tradeoff. The final policy is more capable, but it is also more energetic and slips more because it is doing real work during lateral and combined maneuvers. For sim-to-real transfer, that tradeoff could matter: a more aggressive lateral gait may be harder to execute safely on hardware if actuator limits, latency, or contact modeling differ from simulation. On the other hand, a policy that never truly learns lateral motion is not useful for the real task either. My final design intentionally chose better task completion over minimal energy use.

## Conclusion

Starting from a forward-dominant baseline, I improved stage 2 in two steps: first by changing the command curriculum so that lateral and yaw commands were sampled more deliberately, and second by reshaping the reward so that true lateral motion was rewarded instead of indirectly discouraged. The final policy solved the original problem: it no longer ignored pure `vy` commands, it tracked combined commands well, and it eliminated the falls seen in earlier custom evaluation.

If I had more time, the next direction I would explore is recovering some of the increased energy and slip without losing the lateral gait. The most likely approach would be a mild second-phase fine-tuning pass that keeps the final command curriculum but slightly reintroduces efficiency penalties after lateral tracking has already been learned.
