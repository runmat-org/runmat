//! MATLAB-compatible `step` response builtin for RunMat.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{ComplexTensor, ObjectInstance, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::type_resolvers::step_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "step";
const EPS: f64 = 1.0e-12;
const DEFAULT_SAMPLES: usize = 101;
const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::step")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "step",
    op_kind: GpuOpKind::Custom("control-step-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Step-response simulation runs on the host from transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::step")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "step",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "step simulates a dynamic system and terminates numeric fusion chains.",
};

fn step_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "step",
    category = "control",
    summary = "Compute or plot the step response of a SISO transfer-function model.",
    keywords = "step,response,control system,transfer function,tf",
    sink = true,
    suppress_auto_output = true,
    type_resolver(step_type),
    builtin_path = "crate::builtins::control::step"
)]
async fn step_builtin(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(step_error(
            "step: expected step(sys), step(sys, tFinal), or step(sys, t)",
        ));
    }

    let sys = crate::gather_if_needed_async(&sys).await?;
    let mut rest_host = Vec::with_capacity(rest.len());
    for arg in &rest {
        rest_host.push(crate::gather_if_needed_async(arg).await?);
    }
    let model = TransferFunction::from_value(sys)?;
    let time = TimeSpec::parse(rest_host.first(), model.sample_time)?;
    let eval = evaluate_step(&model, time)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        plot_response(&eval).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            plot_response(&eval).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.y_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs()?,
        ));
    }

    eval.y_value()
}

#[derive(Clone, Debug)]
struct TransferFunction {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    sample_time: f64,
}

impl TransferFunction {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(step_error("step: expected a tf object"));
        };
        if !object.is_class("tf") {
            return Err(step_error(format!(
                "step: expected a tf object, got {}",
                object.class_name
            )));
        }

        let numerator = property_coefficients(&object, "Numerator")?;
        let denominator = property_coefficients(&object, "Denominator")?;
        let sample_time = property_scalar(&object, "Ts")?;
        if !sample_time.is_finite() || sample_time < 0.0 {
            return Err(step_error(
                "step: tf sample time must be finite and non-negative",
            ));
        }

        let numerator = trim_leading_zeros(numerator);
        let denominator = trim_leading_zeros(denominator);
        if denominator.is_empty() {
            return Err(step_error("step: denominator coefficients cannot be empty"));
        }
        if denominator[0].abs() <= EPS {
            return Err(step_error(
                "step: leading denominator coefficient must be non-zero",
            ));
        }
        if numerator.len().saturating_sub(1) > denominator.len().saturating_sub(1) {
            return Err(step_error(
                "step: improper transfer functions are not supported yet",
            ));
        }

        Ok(Self {
            numerator,
            denominator,
            sample_time,
        })
    }

    fn normalized(&self) -> (Vec<f64>, Vec<f64>) {
        let leading = self.denominator[0];
        let den = self
            .denominator
            .iter()
            .map(|value| value / leading)
            .collect::<Vec<_>>();
        let num = self
            .numerator
            .iter()
            .map(|value| value / leading)
            .collect::<Vec<_>>();
        (num, den)
    }
}

fn property_coefficients(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    let value = object
        .properties
        .get(name)
        .ok_or_else(|| step_error(format!("step: tf object is missing {name}")))?;
    match value {
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        Value::ComplexTensor(tensor) => real_complex_coefficients(tensor, name),
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        other => Err(step_error(format!(
            "step: tf {name} coefficients must be numeric, got {other:?}"
        ))),
    }
}

fn real_complex_coefficients(tensor: &ComplexTensor, name: &str) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::with_capacity(tensor.data.len());
    for &(re, im) in &tensor.data {
        if im.abs() > EPS {
            return Err(step_error(format!(
                "step: complex tf {name} coefficients are not supported yet"
            )));
        }
        out.push(re);
    }
    Ok(out)
}

fn property_scalar(object: &ObjectInstance, name: &str) -> BuiltinResult<f64> {
    let value = object
        .properties
        .get(name)
        .ok_or_else(|| step_error(format!("step: tf object is missing {name}")))?;
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        other => Err(step_error(format!(
            "step: tf {name} property must be a scalar, got {other:?}"
        ))),
    }
}

#[derive(Clone, Debug)]
enum TimeSpec {
    Auto,
    FinalTime(f64),
    Vector(Vec<f64>),
}

impl TimeSpec {
    fn parse(value: Option<&Value>, sample_time: f64) -> BuiltinResult<Self> {
        let Some(value) = value else {
            return Ok(Self::Auto);
        };
        match value {
            Value::Num(n) => Self::final_time(*n),
            Value::Int(i) => Self::final_time(i.to_f64()),
            Value::Tensor(tensor) => {
                ensure_time_vector_shape(&tensor.shape)?;
                if tensor.data.len() == 1 {
                    return Self::final_time(tensor.data[0]);
                }
                Self::vector(tensor.data.clone(), sample_time)
            }
            other => Err(step_error(format!(
                "step: time input must be a scalar final time or numeric vector, got {other:?}"
            ))),
        }
    }

    fn final_time(value: f64) -> BuiltinResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(step_error(
                "step: final time must be a positive finite scalar",
            ));
        }
        Ok(Self::FinalTime(value))
    }

    fn vector(values: Vec<f64>, sample_time: f64) -> BuiltinResult<Self> {
        validate_time_vector(&values)?;
        if sample_time > 0.0 {
            for &t in &values {
                let k = (t / sample_time).round();
                if (t - k * sample_time).abs() > 1.0e-8 * sample_time.max(1.0) {
                    return Err(step_error(
                        "step: discrete-time sample vector must align with the model sample time",
                    ));
                }
            }
        }
        Ok(Self::Vector(values))
    }
}

fn ensure_time_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(step_error("step: time input must be a vector"))
    }
}

fn validate_time_vector(values: &[f64]) -> BuiltinResult<()> {
    if values.is_empty() {
        return Err(step_error("step: time vector must not be empty"));
    }
    let mut previous = None;
    for &value in values {
        if !value.is_finite() || value < 0.0 {
            return Err(step_error(
                "step: time vector values must be finite and non-negative",
            ));
        }
        if let Some(prev) = previous {
            if value < prev {
                return Err(step_error("step: time vector must be nondecreasing"));
            }
        }
        previous = Some(value);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct StepEval {
    y: Vec<f64>,
    t: Vec<f64>,
}

impl StepEval {
    fn y_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.y.clone())
    }

    fn t_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.t.clone())
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.y_value()?, self.t_value()?])
    }
}

fn evaluate_step(model: &TransferFunction, time: TimeSpec) -> BuiltinResult<StepEval> {
    if model.sample_time > 0.0 {
        evaluate_discrete_step(model, time)
    } else {
        evaluate_continuous_step(model, time)
    }
}

fn evaluate_continuous_step(model: &TransferFunction, time: TimeSpec) -> BuiltinResult<StepEval> {
    let t = continuous_time_vector(model, time)?;
    let (num, den) = model.normalized();
    let response = continuous_response(&num, &den, &t)?;
    Ok(StepEval { y: response, t })
}

fn continuous_time_vector(model: &TransferFunction, time: TimeSpec) -> BuiltinResult<Vec<f64>> {
    match time {
        TimeSpec::Auto => Ok(linspace(0.0, automatic_final_time(model), DEFAULT_SAMPLES)),
        TimeSpec::FinalTime(final_time) => Ok(linspace(0.0, final_time, DEFAULT_SAMPLES)),
        TimeSpec::Vector(values) => Ok(values),
    }
}

fn continuous_response(num: &[f64], den: &[f64], t: &[f64]) -> BuiltinResult<Vec<f64>> {
    let order = den.len() - 1;
    if order == 0 {
        let gain = num.last().copied().unwrap_or(0.0) / den[0];
        return Ok(vec![gain; t.len()]);
    }

    let mut padded_num = vec![0.0; order + 1 - num.len()];
    padded_num.extend_from_slice(num);
    let direct = padded_num[0];
    let a = &den[1..];
    let mut c = Vec::with_capacity(order);
    for state_idx in 0..order {
        let coeff_idx = order - state_idx;
        c.push(padded_num[coeff_idx] - direct * den[coeff_idx]);
    }

    let mut state = vec![0.0; order];
    let mut current_t = 0.0;
    let mut response = Vec::with_capacity(t.len());
    for &target_t in t {
        if target_t > current_t {
            integrate_to(&mut state, a, current_t, target_t);
            current_t = target_t;
        }
        response.push(dot(&c, &state) + direct);
    }
    Ok(response)
}

fn integrate_to(state: &mut [f64], a: &[f64], start: f64, end: f64) {
    let duration = end - start;
    if duration <= 0.0 {
        return;
    }
    let steps = ((duration / 0.01).ceil() as usize).clamp(1, 10_000);
    let h = duration / steps as f64;
    for _ in 0..steps {
        rk4_step(state, a, h);
    }
}

fn rk4_step(state: &mut [f64], a: &[f64], h: f64) {
    let k1 = derivative(state, a);
    let s2 = add_scaled(state, &k1, h * 0.5);
    let k2 = derivative(&s2, a);
    let s3 = add_scaled(state, &k2, h * 0.5);
    let k3 = derivative(&s3, a);
    let s4 = add_scaled(state, &k3, h);
    let k4 = derivative(&s4, a);
    for idx in 0..state.len() {
        state[idx] += h * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]) / 6.0;
    }
}

fn derivative(state: &[f64], a: &[f64]) -> Vec<f64> {
    let order = state.len();
    let mut dx = vec![0.0; order];
    if order > 1 {
        dx[..(order - 1)].copy_from_slice(&state[1..order]);
    }
    let mut last = 1.0;
    for state_idx in 0..order {
        let coeff = a[order - 1 - state_idx];
        last -= coeff * state[state_idx];
    }
    dx[order - 1] = last;
    dx
}

fn add_scaled(state: &[f64], delta: &[f64], scale: f64) -> Vec<f64> {
    state
        .iter()
        .zip(delta)
        .map(|(value, delta)| value + scale * delta)
        .collect()
}

fn evaluate_discrete_step(model: &TransferFunction, time: TimeSpec) -> BuiltinResult<StepEval> {
    let t = discrete_time_vector(model.sample_time, time)?;
    if t.len() > MAX_DISCRETE_SAMPLES {
        return Err(step_error(format!(
            "step: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    let sample_indices = t
        .iter()
        .map(|&value| checked_discrete_sample_index(model.sample_time, value))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let max_k = sample_indices.iter().copied().max().unwrap_or(0);
    let count = max_k
        .checked_add(1)
        .ok_or_else(|| step_error("step: discrete sample index exceeds platform limits"))?;
    let (num, den) = model.normalized();
    let all_y = discrete_response(&num, &den, count)?;
    let y = sample_indices
        .into_iter()
        .map(|idx| {
            all_y
                .get(idx)
                .copied()
                .ok_or_else(|| step_error("step: discrete sample index exceeds response length"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(StepEval { y, t })
}

fn discrete_time_vector(sample_time: f64, time: TimeSpec) -> BuiltinResult<Vec<f64>> {
    match time {
        TimeSpec::Auto => Ok((0..DEFAULT_SAMPLES)
            .map(|idx| idx as f64 * sample_time)
            .collect()),
        TimeSpec::FinalTime(final_time) => {
            let steps = checked_discrete_sample_steps(sample_time, final_time)?;
            Ok((0..=steps).map(|idx| idx as f64 * sample_time).collect())
        }
        TimeSpec::Vector(values) => Ok(values),
    }
}

fn checked_discrete_sample_steps(sample_time: f64, final_time: f64) -> BuiltinResult<usize> {
    let steps = (final_time / sample_time).floor();
    if !steps.is_finite() || steps < 0.0 || steps > usize::MAX as f64 {
        return Err(step_error(
            "step: discrete sample count exceeds platform limits",
        ));
    }
    if steps >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(step_error(format!(
            "step: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    Ok(steps as usize)
}

fn checked_discrete_sample_index(sample_time: f64, time: f64) -> BuiltinResult<usize> {
    let index = (time / sample_time).round();
    if !index.is_finite() || index < 0.0 || index > usize::MAX as f64 {
        return Err(step_error(
            "step: discrete sample index exceeds platform limits",
        ));
    }
    if index >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(step_error(format!(
            "step: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    Ok(index as usize)
}

fn discrete_response(num: &[f64], den: &[f64], count: usize) -> BuiltinResult<Vec<f64>> {
    let order = den.len() - 1;
    let mut padded_num = vec![0.0; order + 1 - num.len()];
    padded_num.extend_from_slice(num);
    let mut y = vec![0.0; count];
    for k in 0..count {
        let mut value = 0.0;
        for (idx, &coeff) in padded_num.iter().enumerate() {
            if k >= idx {
                value += coeff;
            }
        }
        for idx in 1..den.len() {
            if k >= idx {
                value -= den[idx] * y[k - idx];
            }
        }
        y[k] = value;
    }
    Ok(y)
}

async fn plot_response(eval: &StepEval) -> BuiltinResult<()> {
    let t = eval.t_value()?;
    let y = eval.y_value()?;
    if let Err(err) = crate::call_builtin_async("plot", &[t, y]).await {
        if super::is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(err);
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Step Response")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Time")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Amplitude")]).await;
    Ok(())
}

fn automatic_final_time(model: &TransferFunction) -> f64 {
    let (_, den) = model.normalized();
    let poles = polynomial_roots(&den).unwrap_or_default();
    let slowest_decay = poles
        .iter()
        .filter_map(|pole| if pole.re < -EPS { Some(-pole.re) } else { None })
        .fold(f64::INFINITY, f64::min);
    if slowest_decay.is_finite() && slowest_decay > EPS {
        (5.0 / slowest_decay).clamp(1.0, 100.0)
    } else {
        10.0
    }
}

fn polynomial_roots(coeffs: &[f64]) -> BuiltinResult<Vec<Complex64>> {
    let trimmed = trim_leading_zeros(coeffs.to_vec());
    if trimmed.len() <= 1 {
        return Ok(Vec::new());
    }
    if trimmed.len() == 2 {
        return Ok(vec![Complex64::new(-trimmed[1] / trimmed[0], 0.0)]);
    }
    let degree = trimmed.len() - 1;
    let leading = trimmed[0];
    let mut companion = DMatrix::<Complex64>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }
    for (idx, coeff) in trimmed.iter().enumerate().skip(1) {
        companion[(0, idx - 1)] = Complex64::new(-coeff / leading, 0.0);
    }
    let eigenvalues = companion
        .eigenvalues()
        .ok_or_else(|| step_error("step: failed to compute transfer-function poles"))?;
    Ok(eigenvalues.iter().copied().collect())
}

fn trim_leading_zeros(coeffs: Vec<f64>) -> Vec<f64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.abs() > EPS)
        .unwrap_or(coeffs.len());
    coeffs[first_nonzero..].to_vec()
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![end];
    }
    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    let tensor =
        Tensor::new(data, vec![rows, 1]).map_err(|err| step_error(format!("step: {err}")))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ObjectInstance};

    fn tf_object(num: Vec<f64>, den: Vec<f64>, sample_time: f64) -> Value {
        let mut object = ObjectInstance::new("tf".to_string());
        object.properties.insert(
            "Numerator".to_string(),
            Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
        );
        object.properties.insert(
            "Denominator".to_string(),
            Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
        );
        object.properties.insert(
            "Variable".to_string(),
            Value::CharArray(CharArray::new_row(if sample_time > 0.0 {
                "z"
            } else {
                "s"
            })),
        );
        object
            .properties
            .insert("Ts".to_string(), Value::Num(sample_time));
        Value::Object(object)
    }

    fn run_step(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(step_builtin(sys, rest))
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn first_order_continuous_response_matches_closed_form_for_explicit_time() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let time = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 2.0], vec![1, 4]).unwrap());
        let y = tensor_data(run_step(sys, vec![time]).expect("step"));
        for (actual, t) in y.iter().zip([0.0_f64, 0.5, 1.0, 2.0]) {
            let expected = 1.0 - (-t).exp();
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "t={t} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn multi_output_returns_y_then_time() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let _guard = crate::output_count::push_output_count(Some(2));
        let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap());
        let result = run_step(sys, vec![time]).expect("step");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        assert_eq!(tensor_data(outputs[1].clone()), vec![0.0, 1.0]);
        let y = tensor_data(outputs[0].clone());
        assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
    }

    #[test]
    fn single_requested_output_returns_only_response() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let _guard = crate::output_count::push_output_count(Some(1));
        let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        let result = run_step(sys, vec![time]).expect("step");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 1);
        let y = tensor_data(outputs[0].clone());
        assert_eq!(y.len(), 2);
        assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
    }

    #[test]
    fn scalar_final_time_generates_column_time_vector_ending_at_final_time() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_step(sys, vec![Value::Num(5.0)]).expect("step");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let t = tensor_data(outputs[1].clone());
        assert_eq!(t.len(), DEFAULT_SAMPLES);
        assert_eq!(t[0], 0.0);
        assert!((t[t.len() - 1] - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn discrete_response_uses_sample_time_grid() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 0.1);
        let time = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2], vec![1, 3]).unwrap());
        let y = tensor_data(run_step(sys, vec![time]).expect("step"));
        assert_eq!(y, vec![0.0, 1.0, 1.5]);
    }

    #[test]
    fn discrete_final_time_rejects_excessive_sample_count() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0e-6);
        let err = run_step(sys, vec![Value::Num(2.0)]).expect_err("should fail");
        assert!(err.message().contains("more than 1000000 samples"));
    }

    #[test]
    fn discrete_time_vector_rejects_excessive_sample_index() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0);
        let time =
            Value::Tensor(Tensor::new(vec![0.0, MAX_DISCRETE_SAMPLES as f64], vec![1, 2]).unwrap());
        let err = run_step(sys, vec![time]).expect_err("should fail");
        assert!(err.message().contains("more than 1000000 samples"));
    }

    #[test]
    fn rejects_non_tf_input() {
        let err = run_step(Value::Num(1.0), Vec::new()).expect_err("expected error");
        assert!(err.message().contains("expected a tf object"));
    }
}
