# Reward Tuning Assessment - 20260407_1233

## Scope

- Run analyzed: `results_history_CartPole/cart_pole/20260407_1233`
- Summary analysis: `analysis/20260407_1233/analysis_results.xlsx`
- Reward behavior analysis: `analysis/20260407_1233/reward_behavior/`

## Executive read

The new run is directionally better than `20260406_0736`, but it still does not produce real capture of the pendulum.

The good news:

1. The cart no longer receives an artificial early bonus.
2. The selected best-reward episode lasts longer (`1.137 s` vs `0.875 s` in the previous run).
3. The best episode now allows a larger cart excursion (`0.444 m` vs `0.074 m`), which is more compatible with a capture maneuver.
4. Last-500 averages improve modestly:
   - `mean_total_reward_last500`: `-38.73` vs `-39.05`
   - `mean_performance_last500`: `-26.67` vs `-27.93`
   - `mean_abs_pendulum_angle_raw_last500`: `0.3518` vs `0.3702`

The blocking issue:

1. There are still `0` stabilization successes.
2. In the best-reward episodes, `min_abs_angle` is essentially the initial angle.
3. Even the top 15 reward episodes never reduce `|pendulum_angle|` below about `0.154 rad`.
4. Both active bandwidth bonuses are effectively zero in the last 500 episodes.

This means the current shaping is no longer misguiding the cart as strongly as before, but it is still not rewarding the intermediate phase where the cart must move under the pendulum and start reducing angle meaningfully.

## What changed relative to the previous run

### Previous run: 20260406_0736

- Cart got a positive effective loop reward in the best episode because of the cart band bonus.
- Pendulum loop remained strongly negative.
- The cart was over-rewarded for staying near zero too early.

### Current run: 20260407_1233

- `extra_bonus_band_cart_position_mean_last500 = 0.0`
- `extra_bonus_band_pendulum_angle_mean_last500 = 0.0`
- Reward means in last 500:
  - pendulum loop agent reward: about `-0.4247`
  - cart loop agent reward: about `-0.1120`
- Selected best episode loop means:
  - `assigned_reward_pendulum_angle = -0.3879`
  - `assigned_reward_cart_position = -0.0248`

Interpretation:

The asymmetry is smaller and healthier now, but the reward still provides no explicit positive signal for partial pendulum capture. The system is now less wrong, but still too sparse in the phase that matters most.

## Strong findings

### 1. No real capture yet

For the best-reward episode (`episode_id = 1412`):

- initial angle: `0.157 rad`
- minimum absolute angle during the full episode: `0.157 rad`
- final angle: `1.048 rad`
- final cart position: `0.444 m`

So the cart moves more, but the pendulum is never truly brought closer to vertical than the initial condition.

### 2. The pendulum band is too tight for the current learning stage

Current active band:

```yaml
error_pendulum_angle: [-0.015, 0.015]
```

This range is appropriate only after the policy already knows how to capture. At the current stage it never activates, so it contributes nothing.

### 3. Cart error is still too influential relative to the plant objective

Current `L_e` gate for cart:

```yaml
cart_position: {scaled: 1.0, min_factor: 0.35, max_factor: 0.6}
```

This is still larger than the intended reduction proposed in the previous review. Given that the failure mode is still angle-only, the cart should remain secondary until the pendulum starts to be captured.

### 4. The principal gates are not helping in practice

Recorded last-500 means:

- `principal_gate_L_edot_pendulum_angle_mean = 1.0`
- `principal_gate_L_edot_cart_position_mean = 1.0`

So in the data they are effectively fully open across the final learning window. Whether due to source scale or source selection, they are not creating a real phase separation in practice.

## Recommended next tuning

## Option A: stay within the currently active mechanisms

This is the minimum-change next test.

```yaml
reward_base:
  reward_calculation:
    principal_reward:
      weighted_exponential_params:
        features:
          L_e:    {weight: 0.90, scaled: 6.5, setpoint: 0.0}
          L_edot: {weight: 0.07, scaled: 1.8, setpoint: 0.0}
          L_I:    {weight: 0.00, scaled: 1.0, setpoint: 0.0}
          L_u:    {weight: 0.03, scaled: 1.0, setpoint: 0.0}
        feature_gates:
          L_e:
            per_var:
              pendulum_angle: {scaled: 1.0, min_factor: 1.0, max_factor: 1.0}
              cart_position:  {scaled: 1.0, min_factor: 0.20, max_factor: 0.20}
    extra_rewards:
      bonus_approach:
        bandwidth_bonus:
          per_step_band_bonus: 0.004
          max_total_band_bonus:
            pendulum_angle: 3.0
            cart_position: 0.0
          ranges:
            error_pendulum_angle: [-0.14, 0.14]
            error_cart_position: [-0.01, 0.01]
```

### Why Option A

1. The pendulum needs a reward signal before it reaches the fine stabilization zone.
2. `[-0.015, 0.015]` is far too fine for the current training stage.
3. A pendulum band around `+-0.14 rad` is still stricter than the initial `0.157 rad`, so it does not reward the initial state, but it finally rewards real partial progress.
4. The cart bonus should remain at `0.0`.
5. Cart `L_e` should be reduced further from the current effective ceiling (`0.6`) to `0.2`.

## Option B: the recommendation I trust more

The latest run suggests that active-only tuning is close to its limit. If you allow one structural change, activate a capture-oriented dynamic incentive for the pendulum.

Recommended first activation:

```yaml
extra_rewards:
  conditional_approach:
    dynamic_incentive:
      enabled: true
      method: adaptative
      dynamic_incentive_adapt_params:
        pendulum_velocity_direction:
          assign_to: 'pendulum_angle'
          reward_mode: 'directional_dense'
          x: 'error_pendulum_angle'
          y: 'pendulum_velocity_raw'
          x_sp: 0.0
          direction_sign: -1.0
          error_strength: 6.0
          alignment_strength: 8.0
          weight: 0.10
```

### Why Option B

The current principal reward penalizes state magnitude, but it does not explicitly reward the right direction of motion during capture. A directional pendulum-velocity incentive gives the agent credit when the pendulum angular motion is actually reducing the angle, even before it enters a narrow band.

## Recommended decision

If you want a conservative next run:

1. Apply Option A only.

If you want the highest chance of breaking the current plateau:

1. Apply Option A.
2. Also activate the single directional incentive from Option B with a modest weight.

## One extra note outside reward

The policy still shows almost no `maintain` for some cart agents:

- `action_kp_cart_position maintain = 0.36%`
- `action_kd_cart_position maintain = 0.32%`

So if the next run still gives `0` stabilization successes after the reward retuning, the next bottleneck is very likely `delta_gain = 0.2`, not just reward shape.
