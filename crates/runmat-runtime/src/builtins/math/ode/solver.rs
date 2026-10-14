mod math;
mod rhs;
mod runge_kutta;
mod stiff;

use runmat_value::Value;

use crate::BuiltinResult;

use self::math::{clamp_step, remaining_distance, scaled_error_norm, scaled_next_step};
use self::runge_kutta::{step_ode23, step_ode45};
use self::stiff::step_ode15s;
use super::common::{check_cancellation, ode_error, OdeInput, OdeMethod, OdeOptions, OdeResult};

pub(crate) async fn solve_ode(
    name: &str,
    method: OdeMethod,
    function: &Value,
    input: &OdeInput,
    options: &OdeOptions,
) -> BuiltinResult<OdeResult> {
    check_cancellation(name, input.runtime.as_ref())?;
    let start = input.tspan[0];
    let end = input.tspan[input.tspan.len() - 1];
    let direction = if end >= start { 1.0 } else { -1.0 };
    let dense_output = input.tspan.len() == 2;

    let mut t = start;
    let mut y = input.y0.clone();
    let mut history_t = vec![t];
    let mut history_y = vec![y.clone()];
    let mut steps = 0usize;
    let mut h = options
        .initial_step
        .unwrap_or_else(|| ((end - start).abs() * 1.0e-2).max(1.0e-6))
        .abs()
        * direction;

    if let Some(max_step) = options.max_step {
        h = h.abs().min(max_step) * direction;
    }
    let min_step = ((end - start).abs().max(1.0) * 1.0e-14).max(1.0e-12);

    if dense_output {
        while remaining_distance(t, end, direction) > 0.0 {
            check_cancellation(name, input.runtime.as_ref())?;
            if steps >= options.max_steps {
                return Err(ode_error(
                    name,
                    format!("{name}: exceeded maximum step count"),
                ));
            }
            let max_h = remaining_distance(t, end, direction);
            h = clamp_step(h, direction, max_h, options.max_step);
            if h.abs() < min_step {
                return Err(ode_error(
                    name,
                    format!("{name}: step size underflow before reaching final time"),
                ));
            }

            let step = attempt_step(name, method, function, t, &y, h, input, options).await?;
            if step.accepted {
                t += h;
                y = step.y_next;
                history_t.push(t);
                history_y.push(y.clone());
                steps += 1;
                h = scaled_next_step(
                    h,
                    step.error_norm,
                    method.embedded_error_order(),
                    direction,
                    options.max_step,
                );
            } else {
                h = scaled_next_step(
                    h,
                    step.error_norm,
                    method.embedded_error_order(),
                    direction,
                    options.max_step,
                );
                if h.abs() < min_step {
                    return Err(ode_error(
                        name,
                        format!("{name}: step size underflow during error control"),
                    ));
                }
            }
        }
    } else {
        for &target_t in input.tspan.iter().skip(1) {
            while remaining_distance(t, target_t, direction) > 0.0 {
                check_cancellation(name, input.runtime.as_ref())?;
                if steps >= options.max_steps {
                    return Err(ode_error(
                        name,
                        format!("{name}: exceeded maximum step count"),
                    ));
                }
                let max_h = remaining_distance(t, target_t, direction);
                h = clamp_step(h, direction, max_h, options.max_step);
                if h.abs() < min_step {
                    return Err(ode_error(
                        name,
                        format!(
                            "{name}: step size underflow before reaching requested tspan point"
                        ),
                    ));
                }

                let step = attempt_step(name, method, function, t, &y, h, input, options).await?;
                if step.accepted {
                    t += h;
                    y = step.y_next;
                    steps += 1;
                    h = scaled_next_step(
                        h,
                        step.error_norm,
                        method.embedded_error_order(),
                        direction,
                        options.max_step,
                    );
                } else {
                    h = scaled_next_step(
                        h,
                        step.error_norm,
                        method.embedded_error_order(),
                        direction,
                        options.max_step,
                    );
                    if h.abs() < min_step {
                        return Err(ode_error(
                            name,
                            format!("{name}: step size underflow during error control"),
                        ));
                    }
                }
            }

            history_t.push(t);
            history_y.push(y.clone());
        }
    }

    Ok(OdeResult {
        t: history_t,
        y_rows: history_y,
    })
}

struct StepAttempt {
    accepted: bool,
    error_norm: f64,
    y_next: Vec<f64>,
}

async fn attempt_step(
    name: &str,
    method: OdeMethod,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
    options: &OdeOptions,
) -> BuiltinResult<StepAttempt> {
    let (y_next, err) = match method {
        OdeMethod::Ode45 => step_ode45(name, function, t, y, h, input).await?,
        OdeMethod::Ode23 => step_ode23(name, function, t, y, h, input).await?,
        OdeMethod::Ode15s => step_ode15s(name, function, t, y, h, input).await?,
    };
    let error_norm = scaled_error_norm(y, &y_next, &err, options.rel_tol, options.abs_tol);
    Ok(StepAttempt {
        accepted: error_norm <= 1.0 || error_norm == 0.0,
        error_norm,
        y_next,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn step_scaling_uses_embedded_error_order_for_ode45() {
        let h = 0.1;
        let err = 1.0e-2;

        let next = scaled_next_step(h, err, OdeMethod::Ode45.embedded_error_order(), 1.0, None);
        let expected = h * 0.9 * err.powf(-1.0 / 5.0);
        assert!((next - expected).abs() < 1.0e-12);
    }

    #[test]
    fn step_scaling_uses_embedded_error_order_for_ode23() {
        let h = 0.1;
        let err = 1.0e-2;

        let next = scaled_next_step(h, err, OdeMethod::Ode23.embedded_error_order(), 1.0, None);
        let expected = h * 0.9 * err.powf(-1.0 / 3.0);
        assert!((next - expected).abs() < 1.0e-12);
    }

    #[test]
    fn scaled_error_norm_treats_nan_as_infinite_error() {
        let norm = scaled_error_norm(&[1.0], &[f64::NAN], &[f64::NAN], 1.0e-3, 1.0e-6);
        assert!(norm.is_infinite());
    }

    #[test]
    fn ode15s_newton_step_handles_picard_unstable_stiff_decay() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("stiff_decay", 573)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 573);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -1000.0 * y,
                    ))
                })
            },
        )));
        drop(entry);
        let input = OdeInput {
            tspan: vec![0.0, 0.1],
            y0: vec![1.0],
            y_shape: vec![1, 1],
            scalar_state: true,
            extensions: super::super::ode15s::ODE_COMPATIBILITY_EXTENSIONS,
            runtime: None,
        };

        let (next, err) = block_on(runtime.scope(step_ode15s(
            "ode15s",
            &Value::FunctionHandle("stiff_decay".into()),
            0.0,
            &[1.0],
            0.1,
            &input,
        )))
        .unwrap();

        assert_eq!(next.len(), 1);
        assert!(next[0].is_finite());
        assert!(next[0] > 0.0);
        assert!(next[0] < 0.02);
        assert!(err[0].is_finite());
    }
}
