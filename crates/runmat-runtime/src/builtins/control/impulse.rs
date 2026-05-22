//! MATLAB-compatible `impulse` response builtin for supported control models.

use nalgebra::DMatrix;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::control::type_resolvers::impulse_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "impulse";
const TF_CLASS: &str = "tf";
const EPS: f64 = 1.0e-12;
const DEFAULT_POINTS: usize = 100;
const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::impulse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "impulse",
    op_kind: GpuOpKind::Custom("control-impulse-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Control-system response evaluation runs on the host. GPU-resident metadata is gathered before simulation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::impulse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "impulse",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Impulse-response simulation materialises host-side time and output vectors and terminates fusion chains.",
};

fn impulse_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "impulse",
    category = "control",
    summary = "Compute or plot the impulse response of a supported dynamic system model.",
    keywords = "impulse,control system,transfer function,response,tf",
    type_resolver(impulse_type),
    builtin_path = "crate::builtins::control::impulse"
)]
async fn impulse_builtin(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let system = TfSystem::parse(system).await?;
    let time = TimeSpec::parse(&system, &rest).await?;
    let response = evaluate_impulse(&system, time)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            emit_impulse_plot(&response).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![response.y_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![response.y_value()?, response.t_value()?],
        ));
    }

    if crate::output_context::requested_output_count() == Some(0) {
        emit_impulse_plot(&response).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    response.y_value()
}

async fn emit_impulse_plot(response: &ImpulseResponse) -> BuiltinResult<()> {
    if let Err(err) = render_impulse_plot(response).await {
        if is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(err);
    }
    Ok(())
}

fn is_nonfatal_plot_setup_error(err: &RuntimeError) -> bool {
    let lower = err.message().to_ascii_lowercase();
    lower.contains("plotting is unavailable")
        || lower.contains("non-main thread")
        || lower.contains("interactive plotting failed")
}

#[derive(Clone, Debug)]
struct TfSystem {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    sample_time: f64,
}

impl TfSystem {
    async fn parse(value: Value) -> BuiltinResult<Self> {
        let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
        let Value::Object(object) = gathered else {
            return Err(impulse_error(format!(
                "impulse: expected a dynamic system model, got {gathered:?}"
            )));
        };
        if object.class_name != TF_CLASS {
            return Err(impulse_error(format!(
                "impulse: unsupported model class '{}'; only SISO tf objects are currently supported",
                object.class_name
            )));
        }

        let numerator = real_coefficients(property(&object, "Numerator")?, "Numerator")?;
        let denominator = real_coefficients(property(&object, "Denominator")?, "Denominator")?;
        let sample_time = scalar_property(property(&object, "Ts")?, "Ts")?;
        let input_delay = scalar_property(property(&object, "InputDelay")?, "InputDelay")?;
        let output_delay = scalar_property(property(&object, "OutputDelay")?, "OutputDelay")?;
        if !sample_time.is_finite() || sample_time < 0.0 {
            return Err(impulse_error(format!(
                "impulse: Ts must be a finite non-negative scalar, got {sample_time}"
            )));
        }
        if !input_delay.is_finite() || input_delay < 0.0 {
            return Err(impulse_error(format!(
                "impulse: InputDelay must be a finite non-negative scalar, got {input_delay}"
            )));
        }
        if !output_delay.is_finite() || output_delay < 0.0 {
            return Err(impulse_error(format!(
                "impulse: OutputDelay must be a finite non-negative scalar, got {output_delay}"
            )));
        }
        if input_delay.abs() > EPS || output_delay.abs() > EPS {
            return Err(impulse_error(
                "impulse: transfer functions with input or output delays are not supported yet",
            ));
        }

        let numerator = trim_leading_zeros(numerator);
        let denominator = trim_leading_zeros(denominator);
        if denominator.is_empty() {
            return Err(impulse_error(
                "impulse: denominator coefficients cannot be empty",
            ));
        }
        if numerator.is_empty() {
            return Ok(Self {
                numerator,
                denominator,
                sample_time,
            });
        }
        if denominator.len() <= 1 {
            return Err(impulse_error(
                "impulse: static-gain transfer functions do not have a finite impulse-response vector",
            ));
        }
        if numerator.len() >= denominator.len() {
            return Err(impulse_error(
                "impulse: only strictly proper SISO tf models are currently supported",
            ));
        }

        Ok(Self {
            numerator,
            denominator,
            sample_time,
        })
    }

    fn is_discrete(&self) -> bool {
        self.sample_time > 0.0
    }
}

fn property<'a>(
    object: &'a runmat_builtins::ObjectInstance,
    name: &str,
) -> BuiltinResult<&'a Value> {
    object
        .properties
        .get(name)
        .ok_or_else(|| impulse_error(format!("impulse: tf object is missing {name} property")))
}

fn real_coefficients(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector(label, &tensor.shape)?;
            finite_values(label, tensor.data.clone())
        }
        Value::Num(n) => finite_values(label, vec![*n]),
        Value::Int(i) => finite_values(label, vec![i.to_f64()]),
        Value::Bool(b) => finite_values(label, vec![if *b { 1.0 } else { 0.0 }]),
        Value::LogicalArray(logical) => {
            ensure_vector(label, &logical.shape)?;
            finite_values(
                label,
                logical
                    .data
                    .iter()
                    .map(|bit| if *bit == 0 { 0.0 } else { 1.0 })
                    .collect(),
            )
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(impulse_error(
            "impulse: complex-coefficient tf models are not supported yet",
        )),
        other => Err(impulse_error(format!(
            "impulse: {label} must be a real numeric coefficient vector, got {other:?}"
        ))),
    }
}

fn finite_values(label: &str, values: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(impulse_error(format!(
            "impulse: {label} coefficients must be finite"
        )));
    }
    Ok(values)
}

fn ensure_vector(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(impulse_error(format!(
            "impulse: {label} coefficients must be a vector"
        )))
    }
}

fn scalar_property(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        other => Err(impulse_error(format!(
            "impulse: {label} must be a real scalar, got {other:?}"
        ))),
    }
}

fn trim_leading_zeros(values: Vec<f64>) -> Vec<f64> {
    let first = values.iter().position(|value| value.abs() > EPS);
    match first {
        Some(idx) => values[idx..].to_vec(),
        None => Vec::new(),
    }
}

#[derive(Clone, Debug)]
enum TimeSpec {
    Values(Vec<f64>),
}

impl TimeSpec {
    async fn parse(system: &TfSystem, rest: &[Value]) -> BuiltinResult<Self> {
        match rest {
            [] => Ok(Self::Values(default_time_vector(system))),
            [value] => {
                let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
                if let Some(final_time) = scalar_time_from_value(&gathered)? {
                    return Ok(Self::Values(time_vector_from_final_time(
                        system, final_time,
                    )?));
                }
                let vector = time_vector_from_value(gathered)?;
                validate_time_vector(system, &vector)?;
                Ok(Self::Values(vector))
            }
            _ => Err(impulse_error(
                "impulse: expected impulse(sys), impulse(sys, tFinal), or impulse(sys, t)",
            )),
        }
    }
}

fn scalar_time_from_value(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(Some(tensor.data[0])),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(Some(if logical.data[0] == 0 { 0.0 } else { 1.0 }))
        }
        Value::Tensor(_) | Value::LogicalArray(_) => Ok(None),
        _ => Ok(None),
    }
}

fn default_time_vector(system: &TfSystem) -> Vec<f64> {
    if system.is_discrete() {
        (0..DEFAULT_POINTS)
            .map(|idx| idx as f64 * system.sample_time)
            .collect()
    } else {
        linspace(0.0, 10.0, DEFAULT_POINTS)
    }
}

fn time_vector_from_final_time(system: &TfSystem, final_time: f64) -> BuiltinResult<Vec<f64>> {
    if !final_time.is_finite() || final_time < 0.0 {
        return Err(impulse_error(
            "impulse: final time must be a finite non-negative scalar",
        ));
    }
    if system.is_discrete() {
        let count = checked_discrete_sample_count(system, final_time)?;
        Ok((0..count)
            .map(|idx| idx as f64 * system.sample_time)
            .collect())
    } else if final_time == 0.0 {
        Ok(vec![0.0])
    } else {
        Ok(linspace(0.0, final_time, DEFAULT_POINTS))
    }
}

fn checked_discrete_sample_count(system: &TfSystem, final_time: f64) -> BuiltinResult<usize> {
    let samples = final_time / system.sample_time;
    if !samples.is_finite() {
        return Err(impulse_error(
            "impulse: discrete sample count exceeds platform limits",
        ));
    }

    let count = samples.floor() + 1.0;
    if count > usize::MAX as f64 || count > MAX_DISCRETE_SAMPLES as f64 {
        return Err(impulse_error(format!(
            "impulse: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    Ok(count as usize)
}

fn checked_discrete_sample_index(system: &TfSystem, time: f64) -> BuiltinResult<usize> {
    let samples = time / system.sample_time;
    let index = samples.round();
    if !index.is_finite() || index > usize::MAX as f64 {
        return Err(impulse_error(
            "impulse: discrete sample index exceeds platform limits",
        ));
    }
    if index >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(impulse_error(format!(
            "impulse: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    Ok(index as usize)
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / ((count - 1) as f64);
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

fn time_vector_from_value(value: Value) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| impulse_error(format!("impulse: time vector must be numeric: {err}")))?;
    ensure_vector("time", &tensor.shape)?;
    if tensor.data.is_empty() {
        return Err(impulse_error("impulse: time vector cannot be empty"));
    }
    Ok(tensor.data)
}

fn validate_time_vector(system: &TfSystem, values: &[f64]) -> BuiltinResult<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(impulse_error(
            "impulse: time vector values must be finite and non-negative",
        ));
    }
    if values.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(impulse_error(
            "impulse: time vector values must be strictly increasing",
        ));
    }
    if system.is_discrete() {
        for &value in values {
            let samples = value / system.sample_time;
            if (samples - samples.round()).abs() > 1.0e-8 {
                return Err(impulse_error(
                    "impulse: discrete-time vectors must use integer multiples of the sample time",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ImpulseResponse {
    t: Vec<f64>,
    y: Vec<f64>,
    discrete: bool,
}

impl ImpulseResponse {
    fn y_value(&self) -> BuiltinResult<Value> {
        let tensor = Tensor::new(self.y.clone(), vec![self.y.len(), 1])
            .map_err(|err| impulse_error(format!("impulse: {err}")))?;
        Ok(Value::Tensor(tensor))
    }

    fn t_value(&self) -> BuiltinResult<Value> {
        let tensor = Tensor::new(self.t.clone(), vec![self.t.len(), 1])
            .map_err(|err| impulse_error(format!("impulse: {err}")))?;
        Ok(Value::Tensor(tensor))
    }
}

fn evaluate_impulse(system: &TfSystem, time: TimeSpec) -> BuiltinResult<ImpulseResponse> {
    let TimeSpec::Values(t) = time;
    let realization = Realization::from_tf(system)?;
    let y = if system.is_discrete() {
        discrete_response(system, &realization, &t)?
    } else {
        continuous_response(&realization, &t)
    };
    Ok(ImpulseResponse {
        t,
        y,
        discrete: system.is_discrete(),
    })
}

#[derive(Clone, Debug)]
struct Realization {
    a: DMatrix<f64>,
    c: Vec<f64>,
}

impl Realization {
    fn from_tf(system: &TfSystem) -> BuiltinResult<Self> {
        if system.numerator.is_empty() {
            let order = system.denominator.len().saturating_sub(1).max(1);
            return Ok(Self {
                a: DMatrix::zeros(order, order),
                c: vec![0.0; order],
            });
        }
        let leading = system.denominator[0];
        if leading.abs() <= EPS {
            return Err(impulse_error(
                "impulse: denominator leading coefficient must be non-zero",
            ));
        }
        let denominator: Vec<f64> = system
            .denominator
            .iter()
            .map(|value| *value / leading)
            .collect();
        let mut numerator: Vec<f64> = system
            .numerator
            .iter()
            .map(|value| *value / leading)
            .collect();
        let order = denominator.len() - 1;
        while numerator.len() < order {
            numerator.insert(0, 0.0);
        }

        let mut a = DMatrix::<f64>::zeros(order, order);
        for row in 0..order.saturating_sub(1) {
            a[(row, row + 1)] = 1.0;
        }
        for col in 0..order {
            a[(order - 1, col)] = -denominator[order - col];
        }
        let c = numerator.into_iter().rev().collect();
        Ok(Self { a, c })
    }
}

fn continuous_response(realization: &Realization, t: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&time| {
            let exp_at = matrix_exp(&(realization.a.clone() * time));
            dot_c_with_last_column(&realization.c, &exp_at)
        })
        .collect()
}

fn discrete_response(
    system: &TfSystem,
    realization: &Realization,
    t: &[f64],
) -> BuiltinResult<Vec<f64>> {
    if t.len() > MAX_DISCRETE_SAMPLES {
        return Err(impulse_error(format!(
            "impulse: discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"
        )));
    }
    let sample_indices: Vec<usize> = t
        .iter()
        .map(|value| checked_discrete_sample_index(system, *value))
        .collect::<BuiltinResult<_>>()?;
    let max_index = sample_indices.iter().copied().max().unwrap_or(0);
    let order = realization.c.len();
    let value_count = max_index
        .checked_add(1)
        .ok_or_else(|| impulse_error("impulse: discrete sample index exceeds platform limits"))?;
    let mut values = vec![0.0; value_count];
    if order == 0 {
        return Ok(sample_indices.into_iter().map(|idx| values[idx]).collect());
    }

    let mut state = vec![0.0; order];
    state[order - 1] = 1.0;
    let impulse_scale = 1.0 / system.sample_time;
    for k in 1..=max_index {
        values[k] = dot(&realization.c, &state) * impulse_scale;
        state = mat_vec_mul(&realization.a, &state);
    }
    Ok(sample_indices.into_iter().map(|idx| values[idx]).collect())
}

fn dot_c_with_last_column(c: &[f64], matrix: &DMatrix<f64>) -> f64 {
    if c.is_empty() {
        return 0.0;
    }
    let last_col = matrix.ncols() - 1;
    c.iter()
        .enumerate()
        .map(|(row, coeff)| coeff * matrix[(row, last_col)])
        .sum()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn mat_vec_mul(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.nrows()];
    for row in 0..matrix.nrows() {
        let mut acc = 0.0;
        for col in 0..matrix.ncols() {
            acc += matrix[(row, col)] * vector[col];
        }
        out[row] = acc;
    }
    out
}

fn matrix_exp(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let norm = matrix_one_norm(matrix);
    let scale_power = if norm <= 0.5 {
        0usize
    } else {
        norm.log2().ceil().max(0.0) as usize + 1
    };
    let scale = 2.0_f64.powi(scale_power as i32);
    let scaled = matrix / scale;
    let n = matrix.nrows();
    let mut result = DMatrix::<f64>::identity(n, n);
    let mut term = DMatrix::<f64>::identity(n, n);
    for k in 1..=48 {
        term = (&term * &scaled) / (k as f64);
        result += &term;
        if matrix_one_norm(&term) <= 1.0e-14 {
            break;
        }
    }
    for _ in 0..scale_power {
        result = &result * &result;
    }
    result
}

fn matrix_one_norm(matrix: &DMatrix<f64>) -> f64 {
    let mut best = 0.0;
    for col in 0..matrix.ncols() {
        let mut sum = 0.0;
        for row in 0..matrix.nrows() {
            sum += matrix[(row, col)].abs();
        }
        if sum > best {
            best = sum;
        }
    }
    best
}

#[cfg(feature = "plot-core")]
async fn render_impulse_plot(response: &ImpulseResponse) -> BuiltinResult<Value> {
    let t = response.t_value()?;
    let y = response.y_value()?;
    let plot_name = if response.discrete { "stem" } else { "plot" };
    crate::dispatcher::call_builtin_async(plot_name, &[t, y]).await
}

#[cfg(not(feature = "plot-core"))]
async fn render_impulse_plot(_response: &ImpulseResponse) -> BuiltinResult<Value> {
    Ok(Value::Num(f64::NAN))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ObjectInstance};

    fn tf_object(num: Vec<f64>, den: Vec<f64>, ts: f64) -> Value {
        tf_object_with_delays(num, den, ts, 0.0, 0.0)
    }

    fn tf_object_with_delays(
        num: Vec<f64>,
        den: Vec<f64>,
        ts: f64,
        input_delay: f64,
        output_delay: f64,
    ) -> Value {
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
            Value::CharArray(CharArray::new_row(if ts > 0.0 { "z" } else { "s" })),
        );
        object.properties.insert("Ts".to_string(), Value::Num(ts));
        object
            .properties
            .insert("InputDelay".to_string(), Value::Num(input_delay));
        object
            .properties
            .insert("OutputDelay".to_string(), Value::Num(output_delay));
        Value::Object(object)
    }

    fn run_impulse(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(impulse_builtin(system, rest))
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn impulse_first_order_continuous_explicit_time() {
        let sys = tf_object(vec![20.0], vec![1.0, 5.0], 0.0);
        let t = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2], vec![1, 3]).unwrap());
        let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
        let expected = [20.0, 20.0 * (-0.5f64).exp(), 20.0 * (-1.0f64).exp()];
        for (actual, expected) in y.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn impulse_second_order_continuous() {
        let sys = tf_object(vec![1.0], vec![1.0, 3.0, 2.0], 0.0);
        let t = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap());
        let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
        for (actual, time) in y.iter().zip([0.0_f64, 0.5, 1.0]) {
            let expected = (-time).exp() - (-2.0 * time).exp();
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn impulse_multi_output_returns_y_and_t_columns() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let sys = tf_object(vec![20.0], vec![1.0, 5.0], 0.0);
        let t_arg = Value::Tensor(Tensor::new(vec![0.0, 0.1], vec![1, 2]).unwrap());
        let result = run_impulse(sys, vec![t_arg]).expect("impulse");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        match &outputs[0] {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 1]),
            other => panic!("expected y tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![0.0, 0.1]);
            }
            other => panic!("expected t tensor, got {other:?}"),
        }
    }

    #[test]
    fn impulse_zero_output_count_emits_no_values() {
        let _guard = crate::output_count::push_output_count(Some(0));
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let result = run_impulse(sys, Vec::new()).expect("impulse");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert!(outputs.is_empty());
    }

    #[test]
    fn impulse_requested_zero_outputs_emits_no_values() {
        let _guard = crate::output_context::push_output_count(0);
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let result = run_impulse(sys, Vec::new()).expect("impulse");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert!(outputs.is_empty());
    }

    #[test]
    fn impulse_discrete_siso_response() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 0.1);
        let t = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2, 0.3], vec![1, 4]).unwrap());
        let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
        assert_eq!(y.len(), 4);
        assert!((y[0] - 0.0).abs() < 1.0e-12);
        assert!((y[1] - 10.0).abs() < 1.0e-12);
        assert!((y[2] - 5.0).abs() < 1.0e-12);
        assert!((y[3] - 2.5).abs() < 1.0e-12);
    }

    #[test]
    fn impulse_discrete_final_time_rejects_excessive_sample_count() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0e-6);
        let err = run_impulse(sys, vec![Value::Num(2.0)]).expect_err("should fail");
        assert!(err.message().contains("more than 1000000 samples"));
    }

    #[test]
    fn impulse_discrete_time_vector_rejects_excessive_sample_index() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0);
        let t =
            Value::Tensor(Tensor::new(vec![0.0, MAX_DISCRETE_SAMPLES as f64], vec![1, 2]).unwrap());
        let err = run_impulse(sys, vec![t]).expect_err("should fail");
        assert!(err.message().contains("more than 1000000 samples"));
    }

    #[test]
    fn impulse_rejects_unsupported_model_type() {
        let object = ObjectInstance::new("ss".to_string());
        let err = run_impulse(Value::Object(object), Vec::new()).expect_err("should fail");
        assert!(err.message().contains("unsupported model class"));
    }

    #[test]
    fn impulse_rejects_direct_feedthrough_tf() {
        let sys = tf_object(vec![1.0, 1.0], vec![1.0, 2.0], 0.0);
        let err = run_impulse(sys, Vec::new()).expect_err("should fail");
        assert!(err.message().contains("strictly proper"));
    }

    #[test]
    fn impulse_rejects_invalid_time_metadata() {
        let err = run_impulse(tf_object(vec![1.0], vec![1.0, -0.5], -0.1), Vec::new())
            .expect_err("negative sample time should fail");
        assert!(err.message().contains("Ts must be"));

        let err = run_impulse(
            tf_object_with_delays(vec![1.0], vec![1.0, 5.0], 0.0, f64::NAN, 0.0),
            Vec::new(),
        )
        .expect_err("NaN input delay should fail");
        assert!(err.message().contains("InputDelay must be"));

        let err = run_impulse(
            tf_object_with_delays(vec![1.0], vec![1.0, 5.0], 0.0, 0.0, -1.0),
            Vec::new(),
        )
        .expect_err("negative output delay should fail");
        assert!(err.message().contains("OutputDelay must be"));
    }
}
