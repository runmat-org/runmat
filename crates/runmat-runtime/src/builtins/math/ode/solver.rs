mod math;
mod dense;
mod rhs;
mod runge_kutta;
mod stiff;

use runmat_value::Value;

use crate::BuiltinResult;

use self::dense::{RungeKuttaStep, StepInterpolant};
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
    let refine = if dense_output {
        options.refine.unwrap_or_else(|| method.default_refine())
    } else {
        1
    };
    if refine > 1 && method == OdeMethod::Ode15s {
        let mut builder = crate::build_runtime_error(
            format!("{name}: Refine greater than 1 requires a continuous extension not implemented for this solver"),
        ).with_builtin(name).with_source(UnsupportedOutputRefinement);
        if let Some(identifier) = runmat_builtins::ODE15S_ERROR_INVALID_ARGUMENT.identifier {
            builder = builder.with_identifier(identifier);
        }
        return Err(builder.build());
    }
    let scalars = input
        .y0
        .len()
        .checked_mul(refine)
        .ok_or_else(|| output_geometry_error(name, input))?;
    std::alloc::Layout::array::<f64>(scalars)
        .and_then(|_| std::alloc::Layout::array::<Vec<f64>>(refine))
        .map_err(|_| output_geometry_error(name, input))?;

    let mut t = start;
    let mut y = input.y0.clone();
    let mut history_t = vec![t];
    let mut history_y = vec![y.clone()];
    let mut steps = 0usize;
    let mut output_cooperation = input
        .runtime
        .as_ref()
        .map(|runtime| runtime.cpu_cooperation());
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

            let next_t = if h.abs() == max_h { end } else { t + h };
            validate_output_times(name, t, next_t, h, refine, direction)?;

            let step = attempt_step(name, method, function, t, &y, h, input, options).await?;
            if step.accepted {
                reserve_rows(name, input, &mut history_t, &mut history_y, refine)?;
                if refine > 1 {
                    let interpolant = step
                        .interpolant
                        .as_ref()
                        .expect("the selected RK method supplies its continuous extension");
                    for sample in 1..refine {
                        check_cancellation(name, input.runtime.as_ref())?;
                        let theta = sample as f64 / refine as f64;
                        let sample_t = t + h * theta;
                        let previous_t =
                            *history_t.last().expect("history contains the initial time");
                        if (sample_t - previous_t) * direction <= 0.0
                            || (next_t - sample_t) * direction <= 0.0
                        {
                            return Err(output_time_error(name));
                        }
                        let row = interpolant
                            .evaluate(name, input.runtime.as_ref(), &y, &step.y_next, h, theta)
                            .await?;
                        history_t.push(sample_t);
                        history_y.push(row);
                        // Retain this budget across rows, including scalar
                        // states whose individual evaluations never exhaust a
                        // component quantum on their own.
                        if let Some(cooperation) = &mut output_cooperation {
                            cooperation.checkpoint(1).await.map_err(|source| {
                                source.into_runtime_error_in_context(input.runtime.as_ref())
                            })?;
                        }
                    }
                }
                t = next_t;
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
                    t = if h.abs() == max_h { target_t } else { t + h };
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
    interpolant: Option<StepInterpolant>,
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
    let dense =
        input.tspan.len() == 2 && options.refine.unwrap_or_else(|| method.default_refine()) > 1;
    let step = match method {
        OdeMethod::Ode45 => step_ode45(name, function, t, y, h, input, dense).await?,
        OdeMethod::Ode23 => step_ode23(name, function, t, y, h, input, dense).await?,
        OdeMethod::Ode15s => {
            let (y_next, error) = step_ode15s(name, function, t, y, h, input).await?;
            RungeKuttaStep {
                y_next,
                error,
                interpolant: None,
            }
        }
    };
    let error_norm = scaled_error_norm(
        y,
        &step.y_next,
        &step.error,
        options.rel_tol,
        options.abs_tol,
    );
    Ok(StepAttempt {
        accepted: error_norm <= 1.0 || error_norm == 0.0,
        error_norm,
        y_next: step.y_next,
        interpolant: step.interpolant,
    })
}

#[derive(Debug)]
pub(super) struct UnsupportedOutputRefinement;
impl std::fmt::Display for UnsupportedOutputRefinement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("this solver has no implemented continuous output extension")
    }
}
impl std::error::Error for UnsupportedOutputRefinement {}

#[derive(Debug)]
struct OutputTimeResolution;
impl std::fmt::Display for OutputTimeResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("requested output times are not distinct at floating-point precision")
    }
}
impl std::error::Error for OutputTimeResolution {}

fn output_time_error(name: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "{name}: requested Refine output times are not representably distinct"
    ))
    .with_builtin(name)
    .with_source(OutputTimeResolution)
    .build()
}

fn validate_output_times(
    name: &str,
    start: f64,
    end: f64,
    h: f64,
    refine: usize,
    direction: f64,
) -> BuiltinResult<()> {
    // Reject impossible spacing before stage/output allocation. Every interior
    // point is also checked when emitted; never duplicate, skip or perturb it.
    if (end - start) * direction <= 0.0 {
        return Err(output_time_error(name));
    }
    if refine > 1 {
        let first = start + h / refine as f64;
        let last = start + h * ((refine - 1) as f64 / refine as f64);
        if (first - start) * direction <= 0.0 || (end - last) * direction <= 0.0 {
            return Err(output_time_error(name));
        }
    }
    Ok(())
}

fn output_geometry_error(name: &str, input: &OdeInput) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        input.runtime.as_ref(),
        "RunMat:Placement:ResourceOverflow",
        format!("{name}: refined output geometry overflowed"),
        crate::context::host_allocation::HostAllocationError::Overflow,
    )
}

fn reserve_rows(
    name: &str,
    input: &OdeInput,
    times: &mut Vec<f64>,
    rows: &mut Vec<Vec<f64>>,
    additional: usize,
) -> BuiltinResult<()> {
    times
        .len()
        .checked_add(additional)
        .ok_or_else(|| output_geometry_error(name, input))?;
    let allocation_error = |source| {
        crate::build_runtime_error(format!("{name}: output history allocation failed"))
            .with_builtin(name)
            .with_identifier("RunMat:Placement:ResourceLimit")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    };
    times.try_reserve(additional).map_err(allocation_error)?;
    rows.try_reserve(additional).map_err(allocation_error)?;
    Ok(())
}

#[cfg(test)]
#[path = "solver/refine_tests.rs"]
mod refine_tests;

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
        )))
        .expect("call-port scope");
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
