//! Step-response metrics for SISO transfer-function models and sampled responses.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{control_error, scalar_f64, scalar_text, EPS};
use crate::builtins::control::type_resolvers::stepinfo_type;
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "stepinfo";
const DEFAULT_SETTLING_THRESHOLD: f64 = 0.02;

const STEPINFO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Step-response metrics struct.",
}];
const STEPINFO_INPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const STEPINFO_PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response samples.",
};
const STEPINFO_PARAM_T: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Time samples.",
};
const STEPINFO_PARAM_YFINAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yfinal",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Steady-state value.",
};
const STEPINFO_INPUT_Y_T: [BuiltinParamDescriptor; 2] = [STEPINFO_PARAM_Y, STEPINFO_PARAM_T];
const STEPINFO_INPUT_Y_T_FINAL: [BuiltinParamDescriptor; 3] =
    [STEPINFO_PARAM_Y, STEPINFO_PARAM_T, STEPINFO_PARAM_YFINAL];
const STEPINFO_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(sys)",
        inputs: &STEPINFO_INPUT_SYS,
        outputs: &STEPINFO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(y, t)",
        inputs: &STEPINFO_INPUT_Y_T,
        outputs: &STEPINFO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "info = stepinfo(y, t, yfinal)",
        inputs: &STEPINFO_INPUT_Y_T_FINAL,
        outputs: &STEPINFO_OUTPUT,
    },
];
const STEPINFO_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor {
        code: "RM.STEPINFO.INVALID_ARGUMENT",
        identifier: Some("RunMat:stepinfo:InvalidArgument"),
        when: "Inputs do not match supported stepinfo invocation forms.",
        message: "stepinfo: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.STEPINFO.INVALID_DATA",
        identifier: Some("RunMat:stepinfo:InvalidData"),
        when: "Response, time, or final-value data is malformed.",
        message: "stepinfo: invalid response data",
    },
    BuiltinErrorDescriptor {
        code: "RM.STEPINFO.INVALID_SYSTEM",
        identifier: Some("RunMat:stepinfo:InvalidSystem"),
        when: "System input is not a supported SISO tf object.",
        message: "stepinfo: invalid system",
    },
    BuiltinErrorDescriptor {
        code: "RM.STEPINFO.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:stepinfo:UnsupportedModel"),
        when: "System model form is unsupported.",
        message: "stepinfo: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.STEPINFO.INTERNAL",
        identifier: Some("RunMat:stepinfo:Internal"),
        when: "Response simulation or metric assembly failed.",
        message: "stepinfo: internal error",
    },
];
pub const STEPINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STEPINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STEPINFO_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::stepinfo")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stepinfo",
    op_kind: GpuOpKind::Custom("control-stepinfo"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "stepinfo gathers sampled response data and computes scalar metrics on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::stepinfo")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stepinfo",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "stepinfo computes response metrics and is not fused.",
};

#[runtime_builtin(
    name = "stepinfo",
    category = "control",
    summary = "Compute step-response metrics from SISO models or sampled responses.",
    keywords = "stepinfo,step response,rise time,settling time,overshoot,control system",
    type_resolver(stepinfo_type),
    descriptor(crate::builtins::control::stepinfo::STEPINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::control::stepinfo"
)]
async fn stepinfo_builtin(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let gathered_input = crate::dispatcher::gather_if_needed_async(&input).await?;
    if matches!(gathered_input, Value::Object(_)) {
        if !rest.is_empty() && !is_name_value_start(&rest[0]) {
            return Err(stepinfo_error(
                "stepinfo(sys) does not accept positional response data",
            ));
        }
        let threshold = parse_options(&rest, 0)?;
        let response = crate::call_builtin_async_with_outputs(
            "step",
            std::slice::from_ref(&gathered_input),
            2,
        )
        .await?;
        let (y, t) = output_list_two(response)?;
        let y = numeric_vector(y, "y").await?;
        let t = numeric_vector(t, "t").await?;
        let gain = crate::call_builtin_async("dcgain", &[gathered_input]).await?;
        let y_final = match output_complex_scalar_from_value(gain)? {
            Some(value) => value,
            None => *y
                .last()
                .ok_or_else(|| stepinfo_error("response vector cannot be empty"))?,
        };
        return metrics_to_struct(compute_metrics(&y, &t, y_final, threshold)?);
    }

    let y = numeric_vector(gathered_input, "y").await?;
    let (t, y_final, option_start) = parse_sampled_response_tail(&y, &rest).await?;
    let threshold = parse_options(&rest, option_start)?;
    metrics_to_struct(compute_metrics(&y, &t, y_final, threshold)?)
}

async fn parse_sampled_response_tail(
    y: &[f64],
    rest: &[Value],
) -> BuiltinResult<(Vec<f64>, f64, usize)> {
    if rest.is_empty() || is_name_value_start(&rest[0]) {
        let t = (0..y.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let y_final = *y
            .last()
            .ok_or_else(|| stepinfo_error("response vector cannot be empty"))?;
        return Ok((t, y_final, 0));
    }
    let t = numeric_vector(rest[0].clone(), "t").await?;
    let mut option_start = 1;
    let y_final = if rest.len() > 1 && !is_name_value_start(&rest[1]) {
        let gathered = crate::dispatcher::gather_if_needed_async(&rest[1]).await?;
        option_start = 2;
        scalar_f64(&gathered, "yfinal", BUILTIN_NAME)?
    } else {
        *y.last()
            .ok_or_else(|| stepinfo_error("response vector cannot be empty"))?
    };
    Ok((t, y_final, option_start))
}

fn parse_options(rest: &[Value], start: usize) -> BuiltinResult<f64> {
    if rest.len() == start {
        return Ok(DEFAULT_SETTLING_THRESHOLD);
    }
    if !(rest.len() - start).is_multiple_of(2) {
        return Err(stepinfo_error("name-value options must come in pairs"));
    }
    let mut threshold = DEFAULT_SETTLING_THRESHOLD;
    let mut idx = start;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name", BUILTIN_NAME)?;
        match name.trim().to_ascii_lowercase().as_str() {
            "settlingtimethreshold" => {
                let value = scalar_f64(&rest[idx + 1], "SettlingTimeThreshold", BUILTIN_NAME)?;
                if !value.is_finite() || value <= 0.0 || value >= 1.0 {
                    return Err(stepinfo_error(
                        "SettlingTimeThreshold must be a finite scalar between 0 and 1",
                    ));
                }
                threshold = value;
            }
            other => {
                return Err(stepinfo_error(format!(
                    "unsupported stepinfo option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok(threshold)
}

fn is_name_value_start(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

async fn numeric_vector(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(tensor) => {
            ensure_vector_shape(&tensor, label)?;
            finite_vector(tensor.data, label)
        }
        Value::Num(n) => finite_vector(vec![n], label),
        Value::Int(i) => finite_vector(vec![i.to_f64()], label),
        Value::Bool(b) => finite_vector(vec![if b { 1.0 } else { 0.0 }], label),
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .map(|bit| if *bit == 0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            finite_vector(data, label)
        }
        other => Err(stepinfo_error(format!(
            "{label} must be a real numeric vector, got {other:?}"
        ))),
    }
}

fn ensure_vector_shape(tensor: &Tensor, label: &str) -> BuiltinResult<()> {
    let non_unit = tensor.shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(stepinfo_error(format!("{label} must be a vector")))
    }
}

fn finite_vector(values: Vec<f64>, label: &str) -> BuiltinResult<Vec<f64>> {
    if values.is_empty() {
        return Err(stepinfo_error(format!("{label} vector cannot be empty")));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(stepinfo_error(format!("{label} values must be finite")));
    }
    Ok(values)
}

fn output_list_two(value: Value) -> BuiltinResult<(Value, Value)> {
    let Value::OutputList(outputs) = value else {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:stepinfo:Internal",
            "stepinfo: step did not return an output list",
        ));
    };
    if outputs.len() < 2 {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:stepinfo:Internal",
            "stepinfo: step returned too few outputs",
        ));
    }
    Ok((outputs[0].clone(), outputs[1].clone()))
}

fn output_complex_scalar_from_value(value: Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) if n.is_finite() => Ok(Some(n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Complex(re, im) if im.abs() <= EPS && re.is_finite() => Ok(Some(re)),
        Value::Complex(_, _) => Ok(None),
        _ => Ok(None),
    }
}

#[derive(Clone, Debug)]
struct StepMetrics {
    rise_time: f64,
    transient_time: f64,
    settling_time: f64,
    settling_min: f64,
    settling_max: f64,
    overshoot: f64,
    undershoot: f64,
    peak: f64,
    peak_time: f64,
    steady_state_value: f64,
}

fn compute_metrics(
    y: &[f64],
    t: &[f64],
    y_final: f64,
    settling_threshold: f64,
) -> BuiltinResult<StepMetrics> {
    if y.len() != t.len() {
        return Err(stepinfo_error(
            "y and t must have the same number of samples",
        ));
    }
    if y.is_empty() {
        return Err(stepinfo_error("response vector cannot be empty"));
    }
    if !y_final.is_finite() {
        return Err(stepinfo_error("yfinal must be finite"));
    }
    validate_time(t)?;

    let y0 = y[0];
    let amplitude = y_final - y0;
    let amplitude_abs = amplitude.abs();
    let scale = amplitude_abs.max(y_final.abs()).max(1.0);
    let direction = if amplitude >= 0.0 { 1.0 } else { -1.0 };

    let (rise_time, rise_idx) = if amplitude_abs <= EPS {
        (f64::NAN, 0)
    } else {
        let t10 = crossing_time(y, t, y0 + 0.1 * amplitude, direction);
        let (t90, idx90) = crossing_time_with_index(y, t, y0 + 0.9 * amplitude, direction);
        match (t10, t90) {
            (Some(start), Some(end)) => (end - start, idx90.unwrap_or(0)),
            _ => (f64::NAN, 0),
        }
    };

    let settling_band = settling_threshold * scale;
    let mut last_outside = None;
    for (idx, value) in y.iter().enumerate() {
        if (*value - y_final).abs() > settling_band {
            last_outside = Some(idx);
        }
    }
    let settling_time = match last_outside {
        None => t[0],
        Some(idx) if idx + 1 < t.len() => t[idx + 1],
        Some(_) => f64::NAN,
    };

    let settle_window = &y[rise_idx.min(y.len() - 1)..];
    let settling_min = settle_window.iter().copied().fold(f64::INFINITY, f64::min);
    let settling_max = settle_window
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let overshoot = if amplitude_abs <= EPS {
        0.0
    } else if amplitude >= 0.0 {
        ((max_y - y_final) / amplitude_abs * 100.0).max(0.0)
    } else {
        ((y_final - min_y) / amplitude_abs * 100.0).max(0.0)
    };
    let undershoot = if amplitude_abs <= EPS {
        0.0
    } else if amplitude >= 0.0 {
        ((y0 - min_y) / amplitude_abs * 100.0).max(0.0)
    } else {
        ((max_y - y0) / amplitude_abs * 100.0).max(0.0)
    };

    let (peak_idx, peak) = y
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, y[0].abs()));

    Ok(StepMetrics {
        rise_time,
        transient_time: settling_time,
        settling_time,
        settling_min,
        settling_max,
        overshoot,
        undershoot,
        peak,
        peak_time: t[peak_idx],
        steady_state_value: y_final,
    })
}

fn validate_time(t: &[f64]) -> BuiltinResult<()> {
    if t.iter().any(|value| !value.is_finite()) {
        return Err(stepinfo_error("t values must be finite"));
    }
    if t.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(stepinfo_error("t must be nondecreasing"));
    }
    Ok(())
}

fn crossing_time(y: &[f64], t: &[f64], threshold: f64, direction: f64) -> Option<f64> {
    crossing_time_with_index(y, t, threshold, direction).0
}

fn crossing_time_with_index(
    y: &[f64],
    t: &[f64],
    threshold: f64,
    direction: f64,
) -> (Option<f64>, Option<usize>) {
    for idx in 0..y.len() {
        let current = direction * (y[idx] - threshold);
        if current >= -EPS {
            if idx == 0 {
                return (Some(t[0]), Some(0));
            }
            let prev = direction * (y[idx - 1] - threshold);
            let denom = current - prev;
            if denom.abs() <= EPS {
                return (Some(t[idx]), Some(idx));
            }
            let alpha = (-prev / denom).clamp(0.0, 1.0);
            return (Some(t[idx - 1] + alpha * (t[idx] - t[idx - 1])), Some(idx));
        }
    }
    (None, None)
}

fn metrics_to_struct(metrics: StepMetrics) -> BuiltinResult<Value> {
    let mut out = StructValue::new();
    out.insert("RiseTime", Value::Num(metrics.rise_time));
    out.insert("TransientTime", Value::Num(metrics.transient_time));
    out.insert("SettlingTime", Value::Num(metrics.settling_time));
    out.insert("SettlingMin", Value::Num(metrics.settling_min));
    out.insert("SettlingMax", Value::Num(metrics.settling_max));
    out.insert("Overshoot", Value::Num(metrics.overshoot));
    out.insert("Undershoot", Value::Num(metrics.undershoot));
    out.insert("Peak", Value::Num(metrics.peak));
    out.insert("PeakTime", Value::Num(metrics.peak_time));
    out.insert("SteadyStateValue", Value::Num(metrics.steady_state_value));
    Ok(Value::Struct(out))
}

fn stepinfo_error(message: impl Into<String>) -> RuntimeError {
    control_error(BUILTIN_NAME, "RunMat:stepinfo:InvalidData", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn sampled_response_reports_basic_metrics() {
        let y = Value::Tensor(Tensor::new(vec![0.0, 0.5, 0.9, 1.0], vec![1, 4]).unwrap());
        let t = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap());
        let Value::Struct(info) =
            block_on(stepinfo_builtin(y, vec![t, Value::Num(1.0)])).expect("stepinfo")
        else {
            panic!("expected struct");
        };
        assert!(
            matches!(info.fields.get("RiseTime"), Some(Value::Num(v)) if (*v - 1.8).abs() < 1.0e-12)
        );
        assert!(matches!(info.fields.get("Overshoot"), Some(Value::Num(v)) if v.abs() < 1.0e-12));
    }

    #[test]
    fn system_response_form_runs_step_and_dcgain() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Struct(info) = block_on(stepinfo_builtin(sys, Vec::new())).expect("stepinfo")
        else {
            panic!("expected struct");
        };
        assert!(
            matches!(info.fields.get("SteadyStateValue"), Some(Value::Num(v)) if (*v - 1.0).abs() < 1.0e-12)
        );
    }
}
