//! MATLAB-compatible `nyquist` frequency-response builtin for supported control models.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{ObjectInstance, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::type_resolvers::nyquist_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "nyquist";
const TF_CLASS: &str = "tf";
const EPS: f64 = 1.0e-12;
const DEFAULT_FREQUENCY_POINTS: usize = 200;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::nyquist")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nyquist",
    op_kind: GpuOpKind::Custom("control-nyquist-frequency-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Nyquist frequency-response evaluation runs on the host from transfer-function metadata. GPU-resident coefficient inputs are gathered by tf before model construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::nyquist")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nyquist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "nyquist materialises host response vectors and terminates numeric fusion chains.",
};

fn nyquist_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "nyquist",
    category = "control",
    summary = "Compute or plot Nyquist frequency responses of SISO transfer-function models.",
    keywords = "nyquist,frequency response,control system,transfer function,tf",
    sink = true,
    suppress_auto_output = true,
    type_resolver(nyquist_type),
    builtin_path = "crate::builtins::control::nyquist"
)]
async fn nyquist_builtin(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(nyquist_error(
            "nyquist: expected nyquist(sys) or nyquist(sys, w)",
        ));
    }

    let system = TfSystem::parse(system).await?;
    let frequencies = FrequencySpec::parse(&system, rest.first()).await?;
    let response = evaluate_nyquist(&system, frequencies)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            render_nyquist_plot(&response).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![response.re_value()?]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![
                response.re_value()?,
                response.im_value()?,
            ]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            response.outputs()?,
        ));
    }

    if crate::output_context::requested_output_count() == Some(0) {
        render_nyquist_plot(&response).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    response.re_value()
}

#[derive(Clone, Debug)]
struct TfSystem {
    numerator: Vec<Complex64>,
    denominator: Vec<Complex64>,
    sample_time: f64,
    is_real: bool,
}

impl TfSystem {
    async fn parse(value: Value) -> BuiltinResult<Self> {
        let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
        let Value::Object(object) = gathered else {
            return Err(nyquist_error(format!(
                "nyquist: expected a dynamic system model, got {gathered:?}"
            )));
        };
        if object.class_name != TF_CLASS {
            return Err(nyquist_error(format!(
                "nyquist: unsupported model class '{}'; only SISO tf objects are currently supported",
                object.class_name
            )));
        }

        let numerator = coefficients(property(&object, "Numerator")?, "Numerator")?;
        let denominator = coefficients(property(&object, "Denominator")?, "Denominator")?;
        let sample_time = scalar_property(property(&object, "Ts")?, "Ts")?;
        let input_delay = scalar_property(property(&object, "InputDelay")?, "InputDelay")?;
        let output_delay = scalar_property(property(&object, "OutputDelay")?, "OutputDelay")?;

        if !sample_time.is_finite() || sample_time < 0.0 {
            return Err(nyquist_error(format!(
                "nyquist: Ts must be a finite non-negative scalar, got {sample_time}"
            )));
        }
        if !input_delay.is_finite() || input_delay < 0.0 {
            return Err(nyquist_error(format!(
                "nyquist: InputDelay must be a finite non-negative scalar, got {input_delay}"
            )));
        }
        if !output_delay.is_finite() || output_delay < 0.0 {
            return Err(nyquist_error(format!(
                "nyquist: OutputDelay must be a finite non-negative scalar, got {output_delay}"
            )));
        }
        if input_delay.abs() > EPS || output_delay.abs() > EPS {
            return Err(nyquist_error(
                "nyquist: transfer functions with input or output delays are not supported yet",
            ));
        }

        let numerator = trim_leading_zeros(numerator);
        let denominator = trim_leading_zeros(denominator);
        if denominator.is_empty() {
            return Err(nyquist_error(
                "nyquist: denominator coefficients cannot be empty",
            ));
        }
        if denominator[0].norm() <= EPS {
            return Err(nyquist_error(
                "nyquist: leading denominator coefficient must be non-zero",
            ));
        }

        let is_real = numerator
            .iter()
            .chain(&denominator)
            .all(|value| value.im.abs() <= EPS);
        Ok(Self {
            numerator,
            denominator,
            sample_time,
            is_real,
        })
    }

    fn is_discrete(&self) -> bool {
        self.sample_time > 0.0
    }
}

fn property<'a>(object: &'a ObjectInstance, name: &str) -> BuiltinResult<&'a Value> {
    object
        .properties
        .get(name)
        .ok_or_else(|| nyquist_error(format!("nyquist: tf object is missing {name} property")))
}

fn coefficients(value: &Value, label: &str) -> BuiltinResult<Vec<Complex64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector(label, &tensor.shape)?;
            finite_complex_values(
                label,
                tensor
                    .data
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            )
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector(label, &tensor.shape)?;
            finite_complex_values(
                label,
                tensor
                    .data
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect(),
            )
        }
        Value::Num(n) => finite_complex_values(label, vec![Complex64::new(*n, 0.0)]),
        Value::Int(i) => finite_complex_values(label, vec![Complex64::new(i.to_f64(), 0.0)]),
        Value::Bool(b) => {
            finite_complex_values(label, vec![Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)])
        }
        Value::Complex(re, im) => finite_complex_values(label, vec![Complex64::new(*re, *im)]),
        other => Err(nyquist_error(format!(
            "nyquist: {label} must be a numeric coefficient vector, got {other:?}"
        ))),
    }
}

fn finite_complex_values(label: &str, values: Vec<Complex64>) -> BuiltinResult<Vec<Complex64>> {
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(nyquist_error(format!(
            "nyquist: {label} coefficients must be finite"
        )));
    }
    Ok(values)
}

fn ensure_vector(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(nyquist_error(format!(
            "nyquist: {label} coefficients must be a vector"
        )))
    }
}

fn scalar_property(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        other => Err(nyquist_error(format!(
            "nyquist: {label} must be a real scalar, got {other:?}"
        ))),
    }
}

#[derive(Clone, Debug)]
enum FrequencySpec {
    Values(Vec<f64>),
}

impl FrequencySpec {
    async fn parse(system: &TfSystem, value: Option<&Value>) -> BuiltinResult<Self> {
        let Some(value) = value else {
            return Ok(Self::Values(default_frequency_vector(system)));
        };
        let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
        let tensor = frequency_tensor_from_value(gathered)?;
        ensure_vector("frequency", &tensor.shape)?;
        validate_frequency_vector(&tensor.data)?;
        Ok(Self::Values(tensor.data))
    }
}

fn frequency_tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => {
            Tensor::new(vec![n], vec![1, 1]).map_err(|err| nyquist_error(format!("nyquist: {err}")))
        }
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|err| nyquist_error(format!("nyquist: {err}"))),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| nyquist_error(format!("nyquist: {err}"))),
        other => Err(nyquist_error(format!(
            "nyquist: frequency input must be a real numeric vector, got {other:?}"
        ))),
    }
}

fn validate_frequency_vector(values: &[f64]) -> BuiltinResult<()> {
    if values.is_empty() {
        return Err(nyquist_error("nyquist: frequency vector cannot be empty"));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(nyquist_error(
            "nyquist: frequency values must be finite and non-negative",
        ));
    }
    Ok(())
}

fn default_frequency_vector(system: &TfSystem) -> Vec<f64> {
    if system.is_discrete() {
        return open_linspace(
            0.0,
            std::f64::consts::PI / system.sample_time,
            DEFAULT_FREQUENCY_POINTS,
        );
    }

    let mut breakpoints = Vec::new();
    for coeffs in [&system.numerator, &system.denominator] {
        if let Ok(roots) = polynomial_roots(coeffs) {
            breakpoints.extend(
                roots
                    .into_iter()
                    .map(|root| root.norm())
                    .filter(|value| value.is_finite() && *value > EPS),
            );
        }
    }

    if breakpoints.is_empty() {
        return logspace(-2.0, 2.0, DEFAULT_FREQUENCY_POINTS);
    }

    let min_w = breakpoints.iter().copied().fold(f64::INFINITY, f64::min);
    let max_w = breakpoints.iter().copied().fold(0.0, f64::max);
    let start = (min_w / 100.0).max(1.0e-4);
    let stop = (max_w * 100.0).max(start * 10.0);
    logspace(start.log10(), stop.log10(), DEFAULT_FREQUENCY_POINTS)
}

#[derive(Clone, Debug)]
struct NyquistResponse {
    re: Vec<f64>,
    im: Vec<f64>,
    w: Vec<f64>,
    mirror_negative_frequency: bool,
}

impl NyquistResponse {
    fn re_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.re.clone())
    }

    fn im_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.im.clone())
    }

    fn w_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.w.clone())
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.re_value()?, self.im_value()?, self.w_value()?])
    }
}

fn evaluate_nyquist(
    system: &TfSystem,
    frequencies: FrequencySpec,
) -> BuiltinResult<NyquistResponse> {
    let FrequencySpec::Values(w) = frequencies;
    let mut re = Vec::with_capacity(w.len());
    let mut im = Vec::with_capacity(w.len());
    for &frequency in &w {
        let point = if system.is_discrete() {
            let phase = frequency * system.sample_time;
            Complex64::new(phase.cos(), phase.sin())
        } else {
            Complex64::new(0.0, frequency)
        };
        let value = transfer_response(system, point)?;
        re.push(zero_small(value.re));
        im.push(zero_small(value.im));
    }
    Ok(NyquistResponse {
        re,
        im,
        w,
        mirror_negative_frequency: system.is_real,
    })
}

fn transfer_response(system: &TfSystem, point: Complex64) -> BuiltinResult<Complex64> {
    let numerator = polynomial_eval(&system.numerator, point);
    let denominator = polynomial_eval(&system.denominator, point);
    if denominator.norm() <= EPS {
        return Err(nyquist_error(
            "nyquist: frequency response is singular at one or more requested frequencies",
        ));
    }
    Ok(numerator / denominator)
}

fn polynomial_eval(coeffs: &[Complex64], point: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for &coeff in coeffs {
        acc = acc * point + coeff;
    }
    acc
}

fn polynomial_roots(coeffs: &[Complex64]) -> BuiltinResult<Vec<Complex64>> {
    let trimmed = trim_leading_zeros(coeffs.to_vec());
    if trimmed.len() <= 1 {
        return Ok(Vec::new());
    }
    if trimmed.len() == 2 {
        return Ok(vec![-trimmed[1] / trimmed[0]]);
    }

    let degree = trimmed.len() - 1;
    let leading = trimmed[0];
    let mut companion = DMatrix::<Complex64>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }
    for (idx, coeff) in trimmed.iter().enumerate().skip(1) {
        companion[(0, idx - 1)] = -*coeff / leading;
    }
    let eigenvalues = companion
        .eigenvalues()
        .ok_or_else(|| nyquist_error("nyquist: failed to compute transfer-function roots"))?;
    Ok(eigenvalues.iter().copied().collect())
}

fn trim_leading_zeros(values: Vec<Complex64>) -> Vec<Complex64> {
    let first = values.iter().position(|value| value.norm() > EPS);
    match first {
        Some(idx) => values[idx..].to_vec(),
        None => Vec::new(),
    }
}

async fn render_nyquist_plot(response: &NyquistResponse) -> BuiltinResult<()> {
    let mut args = vec![response.re_value()?, response.im_value()?];
    if response.mirror_negative_frequency {
        let negative_im = response.im.iter().map(|value| -*value).collect::<Vec<_>>();
        args.push(response.re_value()?);
        args.push(column_tensor(negative_im)?);
    }

    if let Some((&re0, &im0)) = response.re.first().zip(response.im.first()) {
        args.push(column_tensor(vec![re0])?);
        args.push(column_tensor(vec![im0])?);
        args.push(Value::from("x"));
        if response.mirror_negative_frequency {
            args.push(column_tensor(vec![re0])?);
            args.push(column_tensor(vec![-im0])?);
            args.push(Value::from("o"));
        }
    }

    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if super::is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(err);
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Nyquist Diagram")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    let tensor =
        Tensor::new(data, vec![rows, 1]).map_err(|err| nyquist_error(format!("nyquist: {err}")))?;
    Ok(Value::Tensor(tensor))
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / ((count - 1) as f64);
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

fn open_linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let step = (stop - start) / ((count + 1) as f64);
    (0..count)
        .map(|idx| start + (idx + 1) as f64 * step)
        .collect()
}

fn logspace(start_exp: f64, stop_exp: f64, count: usize) -> Vec<f64> {
    linspace(start_exp, stop_exp, count)
        .into_iter()
        .map(|value| 10.0_f64.powf(value))
        .collect()
}

fn zero_small(value: f64) -> f64 {
    if value.abs() <= EPS {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ComplexTensor};

    fn tf_object(num: Vec<f64>, den: Vec<f64>, ts: f64) -> Value {
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
            .insert("InputDelay".to_string(), Value::Num(0.0));
        object
            .properties
            .insert("OutputDelay".to_string(), Value::Num(0.0));
        Value::Object(object)
    }

    fn run_nyquist(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(nyquist_builtin(system, rest))
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn nyquist_first_order_continuous_explicit_frequency() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let w = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = run_nyquist(sys, vec![w]).expect("nyquist");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let re = tensor_data(outputs[0].clone());
        let im = tensor_data(outputs[1].clone());
        let w_out = tensor_data(outputs[2].clone());
        let expected_re = [1.0, 0.5, 0.2];
        let expected_im = [0.0, -0.5, -0.4];
        for ((actual_re, actual_im), (expected_re, expected_im)) in re
            .iter()
            .zip(&im)
            .zip(expected_re.into_iter().zip(expected_im))
        {
            assert!((actual_re - expected_re).abs() < 1.0e-12);
            assert!((actual_im - expected_im).abs() < 1.0e-12);
        }
        assert_eq!(w_out, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn nyquist_two_outputs_returns_real_and_imaginary_columns() {
        let sys = tf_object(vec![1.0], vec![1.0, 2.0, 1.0], 0.0);
        let w = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_nyquist(sys, vec![w]).expect("nyquist");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        match &outputs[0] {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 1]),
            other => panic!("expected real tensor, got {other:?}"),
        }
        let re = tensor_data(outputs[0].clone());
        let im = tensor_data(outputs[1].clone());
        assert!((re[0] - 1.0).abs() < 1.0e-12);
        assert!(im[0].abs() < 1.0e-12);
        assert!(re[1].abs() < 1.0e-12);
        assert!((im[1] + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn nyquist_discrete_uses_unit_circle_frequency_mapping() {
        let sys = tf_object(vec![1.0], vec![1.0, -0.5], 0.1);
        let w = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_nyquist(sys, vec![w]).expect("nyquist");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let re = tensor_data(outputs[0].clone());
        let im = tensor_data(outputs[1].clone());
        assert!((re[0] - 2.0).abs() < 1.0e-12);
        assert!(im[0].abs() < 1.0e-12);
    }

    #[test]
    fn nyquist_discrete_default_grid_excludes_singular_unit_circle_endpoints() {
        let system = TfSystem {
            numerator: vec![Complex64::new(1.0, 0.0)],
            denominator: vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
            sample_time: 0.1,
            is_real: true,
        };
        let w = default_frequency_vector(&system);
        assert_eq!(w.len(), DEFAULT_FREQUENCY_POINTS);
        assert!(w[0] > 0.0);
        assert!(w[w.len() - 1] < std::f64::consts::PI / system.sample_time);

        let _guard = crate::output_count::push_output_count(Some(3));
        run_nyquist(tf_object(vec![1.0], vec![1.0, -1.0], 0.1), Vec::new())
            .expect("pole at z=1 should not be evaluated at w=0");
        run_nyquist(tf_object(vec![1.0], vec![1.0, 1.0], 0.1), Vec::new())
            .expect("pole at z=-1 should not be evaluated at the Nyquist frequency");
    }

    #[test]
    fn nyquist_statement_form_plots_without_error() {
        let sys = tf_object(vec![1.0], vec![1.0, 2.0, 1.0], 0.0);
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = run_nyquist(sys, Vec::new()).expect("nyquist");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[test]
    fn nyquist_rejects_invalid_frequency_vector() {
        let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
        let w = Value::Tensor(Tensor::new(vec![0.0, f64::INFINITY], vec![1, 2]).unwrap());
        let err = run_nyquist(sys, vec![w]).expect_err("should fail");
        assert!(err.message().contains("frequency values must be finite"));
    }

    #[test]
    fn nyquist_rejects_unsupported_model_type() {
        let object = ObjectInstance::new("ss".to_string());
        let err = run_nyquist(Value::Object(object), Vec::new()).expect_err("should fail");
        assert!(err.message().contains("unsupported model class"));
    }

    #[test]
    fn nyquist_complex_coefficients_are_supported() {
        let mut object = ObjectInstance::new("tf".to_string());
        object.properties.insert(
            "Numerator".to_string(),
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).expect("numerator"),
            ),
        );
        object.properties.insert(
            "Denominator".to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        );
        object.properties.insert(
            "Variable".to_string(),
            Value::CharArray(CharArray::new_row("s")),
        );
        object.properties.insert("Ts".to_string(), Value::Num(0.0));
        object
            .properties
            .insert("InputDelay".to_string(), Value::Num(0.0));
        object
            .properties
            .insert("OutputDelay".to_string(), Value::Num(0.0));

        let w = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_nyquist(Value::Object(object), vec![w]).expect("nyquist");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let re = tensor_data(outputs[0].clone());
        let im = tensor_data(outputs[1].clone());
        assert!((re[0] - 1.0).abs() < 1.0e-12);
        assert!((im[0] - 1.0).abs() < 1.0e-12);
    }
}
