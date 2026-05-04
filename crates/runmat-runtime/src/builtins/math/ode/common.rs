use runmat_builtins::{StructValue, Tensor, Value};

use crate::builtins::math::optim::common::{call_function, lookup_option, value_to_real_vector};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const DEFAULT_REL_TOL: f64 = 1.0e-3;
const DEFAULT_ABS_TOL: f64 = 1.0e-6;
const DEFAULT_MAX_STEPS: usize = 100_000;

#[derive(Clone, Copy)]
pub(crate) enum OdeMethod {
    Ode45,
    Ode23,
    Ode15s,
}

impl OdeMethod {
    fn order(self) -> f64 {
        match self {
            Self::Ode45 => 5.0,
            Self::Ode23 => 3.0,
            Self::Ode15s => 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct OdeOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub initial_step: Option<f64>,
    pub max_step: Option<f64>,
    pub max_steps: usize,
}

#[derive(Clone)]
pub(crate) struct OdeInput {
    pub tspan: Vec<f64>,
    pub y0: Vec<f64>,
    pub y_shape: Vec<usize>,
    pub scalar_state: bool,
}

pub(crate) struct OdeResult {
    pub t: Vec<f64>,
    pub y_rows: Vec<Vec<f64>>,
}

pub(crate) fn ode_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn parse_options(
    name: &str,
    value: Option<&Value>,
) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(ode_error(
            name,
            format!("{name}: options must be a struct, got {other:?}"),
        )),
    }
}

pub(crate) fn ode_options_from_struct(
    name: &str,
    options: Option<&StructValue>,
) -> BuiltinResult<OdeOptions> {
    let rel_tol = option_f64(name, options, "RelTol", DEFAULT_REL_TOL)?;
    let abs_tol = option_f64(name, options, "AbsTol", DEFAULT_ABS_TOL)?;
    if rel_tol <= 0.0 || abs_tol <= 0.0 {
        return Err(ode_error(
            name,
            format!("{name}: RelTol and AbsTol must be positive"),
        ));
    }

    let initial_step = option_optional_f64(name, options, "InitialStep")?;
    if let Some(step) = initial_step {
        if step <= 0.0 {
            return Err(ode_error(
                name,
                format!("{name}: InitialStep must be positive"),
            ));
        }
    }

    let max_step = option_optional_f64(name, options, "MaxStep")?;
    if let Some(step) = max_step {
        if step <= 0.0 {
            return Err(ode_error(name, format!("{name}: MaxStep must be positive")));
        }
    }

    let max_steps = option_f64(name, options, "MaxSteps", DEFAULT_MAX_STEPS as f64)?;
    if max_steps < 1.0 {
        return Err(ode_error(
            name,
            format!("{name}: MaxSteps must be at least 1"),
        ));
    }

    Ok(OdeOptions {
        rel_tol,
        abs_tol,
        initial_step,
        max_step,
        max_steps: max_steps.floor() as usize,
    })
}

pub(crate) async fn parse_ode_input(
    name: &str,
    tspan: Value,
    y0: Value,
) -> BuiltinResult<OdeInput> {
    let tspan_value = crate::dispatcher::gather_if_needed_async(&tspan).await?;
    let tspan = value_to_real_vector(name, tspan_value).await?;
    if tspan.len() < 2 {
        return Err(ode_error(
            name,
            format!("{name}: tspan must contain at least two time points"),
        ));
    }
    if tspan.windows(2).any(|w| w[0] == w[1]) {
        return Err(ode_error(
            name,
            format!("{name}: tspan values must be strictly monotonic"),
        ));
    }
    let forward = tspan[tspan.len() - 1] > tspan[0];
    if tspan.windows(2).any(|w| (w[1] > w[0]) != forward) {
        return Err(ode_error(
            name,
            format!("{name}: tspan values must be strictly monotonic"),
        ));
    }

    let y0_value = crate::dispatcher::gather_if_needed_async(&y0).await?;
    let (y0, y_shape, scalar_state) = match y0_value {
        Value::Num(n) => (vec![n], vec![1, 1], true),
        Value::Int(i) => (vec![i.to_f64()], vec![1, 1], true),
        Value::Bool(b) => (vec![if b { 1.0 } else { 0.0 }], vec![1, 1], true),
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (tensor.data, tensor.shape, false)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (
                logical
                    .data
                    .iter()
                    .map(|v| if *v == 0 { 0.0 } else { 1.0 })
                    .collect(),
                logical.shape,
                false,
            )
        }
        other => {
            return Err(ode_error(
                name,
                format!("{name}: y0 must be real numeric, got {other:?}"),
            ))
        }
    };

    if y0.iter().any(|value| !value.is_finite()) {
        return Err(ode_error(name, format!("{name}: y0 values must be finite")));
    }

    Ok(OdeInput {
        tspan,
        y0,
        y_shape,
        scalar_state,
    })
}

pub(crate) async fn solve_ode(
    name: &str,
    method: OdeMethod,
    function: &Value,
    input: &OdeInput,
    options: &OdeOptions,
) -> BuiltinResult<OdeResult> {
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
                    method.order(),
                    direction,
                    options.max_step,
                );
            } else {
                h = scaled_next_step(
                    h,
                    step.error_norm,
                    method.order(),
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
                        method.order(),
                        direction,
                        options.max_step,
                    );
                } else {
                    h = scaled_next_step(
                        h,
                        step.error_norm,
                        method.order(),
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

async fn step_ode45(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 1.0 / 5.0)]);
    let k2 = eval_rhs(name, function, t + h * (1.0 / 5.0), &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 10.0), &y3, input).await?;

    let y4 = lincomb(
        y,
        h,
        &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h * (4.0 / 5.0), &y4, input).await?;

    let y5 = lincomb(
        y,
        h,
        &[
            (&k1, 19372.0 / 6561.0),
            (&k2, -25360.0 / 2187.0),
            (&k3, 64448.0 / 6561.0),
            (&k4, -212.0 / 729.0),
        ],
    );
    let k5 = eval_rhs(name, function, t + h * (8.0 / 9.0), &y5, input).await?;

    let y6 = lincomb(
        y,
        h,
        &[
            (&k1, 9017.0 / 3168.0),
            (&k2, -355.0 / 33.0),
            (&k3, 46732.0 / 5247.0),
            (&k4, 49.0 / 176.0),
            (&k5, -5103.0 / 18656.0),
        ],
    );
    let k6 = eval_rhs(name, function, t + h, &y6, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[
            (&k1, 35.0 / 384.0),
            (&k3, 500.0 / 1113.0),
            (&k4, 125.0 / 192.0),
            (&k5, -2187.0 / 6784.0),
            (&k6, 11.0 / 84.0),
        ],
    );
    let k7 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 5179.0 / 57600.0),
            (&k3, 7571.0 / 16695.0),
            (&k4, 393.0 / 640.0),
            (&k5, -92097.0 / 339200.0),
            (&k6, 187.0 / 2100.0),
            (&k7, 1.0 / 40.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((y_high, err))
}

async fn step_ode23(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 0.5)]);
    let k2 = eval_rhs(name, function, t + h * 0.5, &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k2, 3.0 / 4.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 4.0), &y3, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[(&k1, 2.0 / 9.0), (&k2, 1.0 / 3.0), (&k3, 4.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 7.0 / 24.0),
            (&k2, 1.0 / 4.0),
            (&k3, 1.0 / 3.0),
            (&k4, 1.0 / 8.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((y_high, err))
}

async fn step_ode15s(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let f_n = eval_rhs(name, function, t, y, input).await?;
    let predictor = lincomb(y, h, &[(&f_n, 1.0)]);
    let mut next = predictor.clone();
    let target_t = t + h;
    for _ in 0..8 {
        let f_next = eval_rhs(name, function, target_t, &next, input).await?;
        let candidate = lincomb(y, h, &[(&f_next, 1.0)]);
        let delta = candidate
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        next = candidate;
        if delta <= 1.0e-9 * (1.0 + max_abs(&next)) {
            break;
        }
    }
    let err = next
        .iter()
        .zip(predictor.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok((next, err))
}

async fn eval_rhs(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Vec<f64>> {
    let y_arg = if input.scalar_state {
        Value::Num(y[0])
    } else {
        Value::Tensor(
            Tensor::new(y.to_vec(), input.y_shape.clone())
                .map_err(|e| ode_error(name, format!("{name}: {e}")))?,
        )
    };
    let value = call_function(function, vec![Value::Num(t), y_arg]).await?;
    let rhs = value_to_real_vector(name, value).await?;
    if rhs.len() != y.len() {
        return Err(ode_error(
            name,
            format!(
                "{name}: derivative output length {} does not match state length {}",
                rhs.len(),
                y.len()
            ),
        ));
    }
    Ok(rhs)
}

fn scaled_error_norm(y: &[f64], y_next: &[f64], err: &[f64], rel_tol: f64, abs_tol: f64) -> f64 {
    y.iter()
        .zip(y_next.iter())
        .zip(err.iter())
        .map(|((yn, yn1), e)| {
            let scale = abs_tol + rel_tol * yn.abs().max(yn1.abs());
            e.abs() / scale.max(1.0e-15)
        })
        .fold(0.0_f64, f64::max)
}

fn lincomb(base: &[f64], h: f64, terms: &[(&[f64], f64)]) -> Vec<f64> {
    let mut out = base.to_vec();
    for (vec, coeff) in terms {
        for (dst, src) in out.iter_mut().zip(vec.iter()) {
            *dst += h * coeff * src;
        }
    }
    out
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn remaining_distance(current: f64, target: f64, direction: f64) -> f64 {
    ((target - current) * direction).max(0.0)
}

fn clamp_step(h: f64, direction: f64, remaining: f64, max_step: Option<f64>) -> f64 {
    let mut mag = h.abs().min(remaining);
    if let Some(max_step) = max_step {
        mag = mag.min(max_step);
    }
    mag * direction
}

fn scaled_next_step(
    h: f64,
    error_norm: f64,
    method_order: f64,
    direction: f64,
    max_step: Option<f64>,
) -> f64 {
    let safety = 0.9;
    let min_scale = 0.2;
    let max_scale = 5.0;
    let exponent = 1.0 / (method_order + 1.0);
    let scale = if error_norm <= 1.0e-12 {
        max_scale
    } else {
        (safety * error_norm.powf(-exponent)).clamp(min_scale, max_scale)
    };
    let mut next = h.abs() * scale;
    if let Some(max_step) = max_step {
        next = next.min(max_step);
    }
    next * direction
}

fn option_f64(
    name: &str,
    options: Option<&StructValue>,
    field: &str,
    default: f64,
) -> BuiltinResult<f64> {
    let Some(options) = options else {
        return Ok(default);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        other => Err(ode_error(
            name,
            format!("{name}: option {field} must be numeric, got {other:?}"),
        )),
    }
}

fn option_optional_f64(
    name: &str,
    options: Option<&StructValue>,
    field: &str,
) -> BuiltinResult<Option<f64>> {
    let Some(options) = options else {
        return Ok(None);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(None);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        other => Err(ode_error(
            name,
            format!("{name}: option {field} must be numeric, got {other:?}"),
        )),
    }
}

pub(crate) fn build_ode_output(name: &str, result: OdeResult) -> BuiltinResult<Value> {
    let t = Tensor::new(result.t.clone(), vec![result.t.len(), 1])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))?;
    let y = rows_to_matrix_value(name, &result.y_rows)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![y]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![t, y],
        ));
    }
    Ok(y)
}

fn rows_to_matrix_value(name: &str, rows: &[Vec<f64>]) -> BuiltinResult<Value> {
    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|e| ode_error(name, format!("{name}: {e}")));
    }
    let row_count = rows.len();
    let col_count = rows[0].len();
    let mut data = vec![0.0; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != col_count {
            return Err(ode_error(
                name,
                format!("{name}: internal solver produced inconsistent row lengths"),
            ));
        }
        for (col_idx, value) in row.iter().enumerate() {
            data[row_idx + col_idx * row_count] = *value;
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))
}
