use super::super::common::OdeInput;
use super::math::{lincomb, max_abs};
use super::rhs::eval_rhs;
use crate::BuiltinResult;
use nalgebra::{DMatrix, DVector};
use runmat_value::Value;

const ODE15S_NEWTON_MAX_ITERS: usize = 8;
const ODE15S_NEWTON_DAMPING_TRIES: usize = 6;

pub(super) async fn step_ode15s(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let f_n = eval_rhs(name, function, t, y, input).await?;
    let predictor = lincomb(y, h, &[(&f_n, 1.0)]);
    let target_t = t + h;

    let Some(full_step) =
        implicit_euler_newton(name, function, target_t, y, h, &predictor, input).await?
    else {
        return Ok((predictor, vec![f64::INFINITY; y.len()]));
    };

    let half_h = h * 0.5;
    let midpoint_t = t + half_h;
    let half_predictor = lincomb(y, half_h, &[(&f_n, 1.0)]);
    let Some(midpoint) = implicit_euler_newton(
        name,
        function,
        midpoint_t,
        y,
        half_h,
        &half_predictor,
        input,
    )
    .await?
    else {
        return Ok((full_step, vec![f64::INFINITY; y.len()]));
    };

    let f_midpoint = eval_rhs(name, function, midpoint_t, &midpoint, input).await?;
    let second_half_predictor = lincomb(&midpoint, half_h, &[(&f_midpoint, 1.0)]);
    let Some(next) = implicit_euler_newton(
        name,
        function,
        target_t,
        &midpoint,
        half_h,
        &second_half_predictor,
        input,
    )
    .await?
    else {
        return Ok((full_step, vec![f64::INFINITY; y.len()]));
    };

    let err = next
        .iter()
        .zip(full_step.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((next, err))
}

async fn implicit_euler_newton(
    name: &str,
    function: &Value,
    target_t: f64,
    y_base: &[f64],
    h: f64,
    initial: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut next = initial.to_vec();
    let (mut residual, mut f_next) =
        implicit_euler_residual(name, function, target_t, y_base, h, &next, input).await?;

    if newton_converged(y_base, &next, &residual) {
        return Ok(Some(next));
    }

    for _ in 0..ODE15S_NEWTON_MAX_ITERS {
        let jacobian =
            implicit_euler_jacobian(name, function, target_t, &next, h, &f_next, input).await?;
        let matrix = DMatrix::from_row_slice(next.len(), next.len(), &jacobian);
        let rhs = -DVector::from_column_slice(&residual);
        let Some(delta) = matrix.lu().solve(&rhs) else {
            return Ok(None);
        };
        if delta.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }

        let residual_norm = max_abs(&residual);
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..ODE15S_NEWTON_DAMPING_TRIES {
            let trial = next
                .iter()
                .zip(delta.iter())
                .map(|(value, step)| value + alpha * step)
                .collect::<Vec<_>>();
            if !all_finite(&trial) {
                alpha *= 0.5;
                continue;
            }

            let (trial_residual, trial_f_next) =
                implicit_euler_residual(name, function, target_t, y_base, h, &trial, input).await?;
            let trial_norm = max_abs(&trial_residual);
            if trial_norm < residual_norm || newton_converged(y_base, &trial, &trial_residual) {
                next = trial;
                residual = trial_residual;
                f_next = trial_f_next;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            return Ok(None);
        }

        let step_norm = delta
            .iter()
            .fold(0.0_f64, |acc, value| acc.max((alpha * value).abs()));
        if newton_converged(y_base, &next, &residual)
            || step_norm <= 1.0e-9 * (1.0 + max_abs(&next))
        {
            return Ok(Some(next));
        }
    }

    Ok(None)
}

async fn implicit_euler_residual(
    name: &str,
    function: &Value,
    target_t: f64,
    y_base: &[f64],
    h: f64,
    y_trial: &[f64],
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let f_trial = eval_rhs(name, function, target_t, y_trial, input).await?;
    let residual = y_trial
        .iter()
        .zip(y_base.iter())
        .zip(f_trial.iter())
        .map(|((trial, base), f)| trial - base - h * f)
        .collect::<Vec<_>>();
    Ok((residual, f_trial))
}

async fn implicit_euler_jacobian(
    name: &str,
    function: &Value,
    target_t: f64,
    y: &[f64],
    h: f64,
    f_base: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Vec<f64>> {
    let n = y.len();
    let mut jacobian = vec![0.0; n * n];

    for col in 0..n {
        let mut perturbed = y.to_vec();
        let step = f64::EPSILON.sqrt() * (y[col].abs() + 1.0);
        perturbed[col] += step;
        let f_perturbed = eval_rhs(name, function, target_t, &perturbed, input).await?;
        for row in 0..n {
            let df_dy = (f_perturbed[row] - f_base[row]) / step;
            jacobian[row * n + col] = if row == col { 1.0 } else { 0.0 } - h * df_dy;
        }
    }

    Ok(jacobian)
}

fn newton_converged(y_base: &[f64], y_next: &[f64], residual: &[f64]) -> bool {
    max_abs(residual) <= 1.0e-10 * (1.0 + max_abs(y_base).max(max_abs(y_next)))
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}
