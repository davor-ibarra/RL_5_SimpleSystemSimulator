# Reward Tuning Assessment - 20260406_0736

## Scope

- Run analyzed: `results_history_CartPole/cart_pole/20260406_0736`
- Summary analysis: `analysis/20260406_0736/analysis_results.xlsx`
- Reward behavior analysis: `analysis/20260406_0736/reward_behavior/`

## Key findings

1. There were `5000` episodes and `0` stabilization successes.
2. In the last 500 episodes, `100%` of the failures were `angle_only_fail`; the cart did not dominate the failures.
3. Global saturation is not the main bottleneck: `sat_ratio_mean_last500 = 0.0432`, `p95 = 0.1263`.
4. The cart loop receives a much friendlier signal than the pendulum loop.
   - Last-500 interval reward mean per cart agent: about `-0.119`.
   - Last-500 interval reward mean per pendulum agent: about `-0.447`.
5. The active bandwidth bonus is only helping the cart.
   - `extra_bonus_band_cart_position_mean = 0.010189`
   - `extra_bonus_band_pendulum_angle_mean = 0.0`
6. The principal gates that should delay derivative/effort penalties are effectively open all the time.
   - `principal_gate_L_edot_pendulum_angle_mean = 1.0`
   - `principal_gate_L_edot_cart_position_mean = 1.0`
7. The selected best-reward episode (`episode_id = 1531`) fails in `0.875 s`.
   - Initial pendulum angle: `0.157 rad`
   - Final pendulum angle: `1.0487 rad`
   - The episode never reduces `|pendulum_angle|` below `0.10 rad`
   - Maximum cart excursion is only `0.0744 m`
8. In the last 500 episodes the learned gains are strongly biased toward the cart loop.
   - `kp_cart ~= 7.99`, `ki_cart ~= 6.17`, `kd_cart ~= 9.69`
   - `kp_pend ~= 2.02`, `ki_pend ~= 2.99`, `kd_pend ~= 3.12`

## Control interpretation

The current shaping is rewarding the cart for staying close to zero before the pendulum has been captured. That is the opposite of the sequence the plant needs:

1. Move the cart assertively to get under the pendulum.
2. Once the angle is already small, start penalizing derivative and effort more strongly.
3. Only after capture, care about fine cart recentering.

Right now the cart band bonus is active from the first interval, while the pendulum band bonus never activates. On top of that, the `L_edot` and `L_u` gates are so wide in squared-error units that they do not create the intended phase separation.

## Recommended active-parameter retuning

### Stage 1: bootstrap capture first

```yaml
reward_base:
  reward_calculation:
    principal_reward:
      weighted_exponential_params:
        features:
          L_e:    {weight: 0.88, scaled: 6.5, setpoint: 0.0}
          L_edot: {weight: 0.08, scaled: 1.8, setpoint: 0.0}
          L_I:    {weight: 0.00, scaled: 1.0, setpoint: 0.0}
          L_u:    {weight: 0.04, scaled: 1.0, setpoint: 0.0}
        feature_gates:
          L_e:
            per_var:
              pendulum_angle: {scaled: 1.0, min_factor: 1.0, max_factor: 1.0}
              cart_position:  {scaled: 1.0, min_factor: 0.35, max_factor: 0.35}
          L_edot:
            per_var:
              pendulum_angle: {scaled: 0.015, min_factor: 0.0, max_factor: 1.0}
              cart_position:  {scaled: 0.004, min_factor: 0.0, max_factor: 1.0}
          L_u:
            per_var:
              pendulum_angle: {scaled: 0.012, min_factor: 0.0, max_factor: 1.0}
              cart_position:  {scaled: 0.0025, min_factor: 0.0, max_factor: 1.0}
    extra_rewards:
      bonus_approach:
        bandwidth_bonus:
          per_step_band_bonus: 0.0025
          max_total_band_bonus:
            pendulum_angle: 2.5
            cart_position: 0.0
          ranges:
            error_pendulum_angle: [-0.12, 0.12]
            error_cart_position: [-0.01, 0.01]
```

## Why these values

1. `L_e` is increased from `0.82` to `0.88` to make state recovery dominate the shaping.
2. `L_edot` and `L_u` are reduced because smoothness and effort should be secondary until capture is happening.
3. The `L_edot` and `L_u` gate scales are reduced by roughly one order of magnitude.
   - Current scales are applied to `*_mean_squared`, so values like `0.22` or `0.28` correspond to raw errors around `sqrt(0.22) ~= 0.47` and `sqrt(0.28) ~= 0.53`, which is far too wide.
   - The proposed scales move those penalties into a near-target region instead of keeping them active during the capture phase.
4. `L_e` for the cart is capped to `0.35` of its current effect because cart error is not the failure mode right now.
5. The cart band bonus is disabled in Stage 1 because it is currently rewarding "stay near zero" from the first interval.
6. The pendulum band is widened to `+-0.12 rad` so the loop can start receiving extra reward once it truly improves from the initial `0.157 rad`, without paying a reward at the initial condition itself.

## Stage 2 after first stabilization successes appear

Once the run starts producing non-zero stabilization successes, tighten the shaping back toward final regulation:

```yaml
bandwidth_bonus:
  max_total_band_bonus:
    pendulum_angle: 2.0
    cart_position: 0.2
  ranges:
    error_pendulum_angle: [-0.05, 0.05]
    error_cart_position: [-0.02, 0.02]
```

The idea is to reintroduce a small cart-centering incentive only after the pendulum capture basin has been learned.

## One non-reward note

If the next run still shows `0` stabilization successes after the Stage-1 reward retuning, the next bottleneck is likely not reward scale but action granularity.

- Current `delta_gain = 0.2`
- Last-500 maintain usage is extremely low for `ki_cart_position` and `kd_cart_position`

At that point the next test should reduce `delta_gain` to `0.05` or `0.1`, but that is outside the active reward parameters requested here.
