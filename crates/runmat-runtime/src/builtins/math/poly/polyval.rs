//! MATLAB-compatible `polyval` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex64;
#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::PolyfitStatField;
#[cfg(test)]
use runmat_builtins::POLYVAL_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, POLYVAL_ERROR_INVALID_ARGUMENT, POLYVAL_ERROR_INVALID_INPUT,
    POLYVAL_INTEGER_COEFFICIENTS_EXTENSION, POLYVAL_INTEGER_OPTIONS_EXTENSION,
    POLYVAL_INTEGER_POINTS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, LogicalArray, Tensor, Value};

use super::normalization;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "polyval";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::polyval")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polyval",
    op_kind: GpuOpKind::Custom("polyval"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("polyval")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses provider-level Horner kernels for real coefficients/inputs; falls back to host evaluation (with upload) for complex or prediction-interval paths.",
};

fn polyval_error(message: impl Into<String>) -> RuntimeError {
    polyval_error_with(message, &POLYVAL_ERROR_INVALID_INPUT)
}

fn polyval_argument_error(message: impl Into<String>) -> RuntimeError {
    polyval_error_with(message, &POLYVAL_ERROR_INVALID_ARGUMENT)
}

fn polyval_error_with(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::polyval")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polyval",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Acts as a fusion sink; real-valued workloads stay on device, while complex/delta paths gather to the host.",
};

#[runtime_builtin(
    name = "polyval",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::poly::polyval"
)]
async fn polyval_builtin(
    p: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        let eval = evaluate_view(p, x, rest.view(), out_count >= 2).await?;
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        let mut outputs = vec![eval.value()];
        if out_count >= 2 {
            outputs.push(eval.delta()?);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    let eval = evaluate_view(p, x, rest.view(), false).await?;
    Ok(eval.value())
}

/// Evaluate `polyval`, optionally computing the prediction interval.
pub async fn evaluate(
    coefficients: Value,
    points: Value,
    rest: &[Value],
    want_delta: bool,
) -> BuiltinResult<PolyvalEval> {
    evaluate_view(
        coefficients,
        points,
        crate::call::FunctionArgumentView::from_slice(rest),
        want_delta,
    )
    .await
}

async fn evaluate_view(
    coefficients: Value,
    points: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    want_delta: bool,
) -> BuiltinResult<PolyvalEval> {
    // Arity and an absent/known-empty statistics argument are decidable from
    // the public call without reading any resident numerical payload.
    if rest.len() > 2 {
        return Err(polyval_argument_error("polyval: too many input arguments"));
    }
    if want_delta && (rest.is_empty() || rest.first().is_some_and(is_known_empty_option)) {
        return Err(polyval_argument_error(
            "polyval: S input (structure returned by polyfit) is required for delta output",
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&coefficients, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&points, BUILTIN_NAME)?;
    for option in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(option, BUILTIN_NAME)?;
    }
    let (preparsed_mu, preparsed_stats) = preflight_host_options(rest, &coefficients).await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &coefficients,
        &POLYVAL_INTEGER_COEFFICIENTS_EXTENSION,
        BUILTIN_NAME,
        "coefficient",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &points,
        &POLYVAL_INTEGER_POINTS_EXTENSION,
        BUILTIN_NAME,
        "query point",
    )
    .await?;
    for option in rest {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            option,
            &POLYVAL_INTEGER_OPTIONS_EXTENSION,
            BUILTIN_NAME,
            "fit-option",
        )
        .await?;
    }
    let options = parse_option_values(rest).await?;

    let residency = resident::Request::new(&points, &coefficients, rest)?;
    let coeff_clone = coefficients.clone();
    let points_clone = points.clone();

    let coeff_was_gpu = matches!(coefficients, Value::GpuTensor(_));
    let (coeffs, coeff_real) = convert_coefficients(coeff_clone).await?;

    let (mut inputs, prefer_gpu_points) = convert_points(points_clone).await?;
    let prefer_gpu_output = prefer_gpu_points || coeff_was_gpu;

    let mu = match (preparsed_mu, options.mu.clone()) {
        (Some(mu), _) => Some(mu),
        (None, Some(mu_value)) => Some(parse_mu(mu_value).await?),
        (None, None) => None,
    };

    if prefer_gpu_output && !want_delta && options.s.is_none() {
        if let Some(value) = residency.try_evaluate(&coeffs, coeff_real, &inputs, mu)? {
            return Ok(PolyvalEval::new(value, None));
        }
    }

    if let Some(mu_val) = mu {
        apply_mu(&mut inputs.data, mu_val)?;
    }

    let stats = match (preparsed_stats, options.s) {
        (Some(stats), _) => Some(stats),
        (None, Some(s_value)) => parse_stats(s_value, coeffs.len()).await?,
        (None, None) => None,
    };

    if want_delta && stats.is_none() {
        return Err(polyval_argument_error(
            "polyval: S input (structure returned by polyfit) is required for delta output",
        ));
    }

    if inputs.data.is_empty() {
        let y = zeros_like(&inputs.shape, prefer_gpu_output, &residency)?;
        let delta = if want_delta {
            Some(zeros_like(&inputs.shape, prefer_gpu_output, &residency)?)
        } else {
            None
        };
        return Ok(PolyvalEval::new(y, delta));
    }

    if coeffs.is_empty() {
        let zeros = zeros_like(&inputs.shape, prefer_gpu_output, &residency)?;
        let delta = if want_delta {
            Some(zeros_like(&inputs.shape, prefer_gpu_output, &residency)?)
        } else {
            None
        };
        return Ok(PolyvalEval::new(zeros, delta));
    }

    let output_real = coeff_real && inputs.all_real;
    let values = evaluate_polynomial(&coeffs, &inputs.data);
    let result_value = finalize_values(
        &values,
        &inputs.shape,
        prefer_gpu_output,
        output_real && values_are_real(&values),
        &residency,
    )?;

    let delta_value = if want_delta {
        let stats = stats.expect("delta requires stats");
        let delta = compute_prediction_interval(&coeffs, &inputs.data, &stats)?;
        let prefer = prefer_gpu_output && stats.is_real;
        Some(finalize_delta(delta, &inputs.shape, prefer, &residency)?)
    } else {
        None
    };

    Ok(PolyvalEval::new(result_value, delta_value))
}

#[path = "polyval/resident.rs"]
mod resident;

/// Result object for polyval evaluation.
#[derive(Debug)]
pub struct PolyvalEval {
    value: Value,
    delta: Option<Value>,
}

impl PolyvalEval {
    fn new(value: Value, delta: Option<Value>) -> Self {
        Self { value, delta }
    }

    /// Primary output (`y`).
    pub fn value(&self) -> Value {
        self.value.clone()
    }

    /// Optional prediction interval (`delta`).
    pub fn delta(&self) -> BuiltinResult<Value> {
        self.delta
            .clone()
            .ok_or_else(|| polyval_argument_error("polyval: delta output not computed"))
    }

    /// Consume into the main value.
    pub fn into_value(self) -> Value {
        self.value
    }

    /// Consume into `(value, delta)` pair.
    pub fn into_pair(self) -> BuiltinResult<(Value, Value)> {
        match self.delta {
            Some(delta) => Ok((self.value, delta)),
            None => Err(polyval_argument_error("polyval: delta output not computed")),
        }
    }
}

#[derive(Clone, Copy)]
struct Mu {
    mean: f64,
    scale: f64,
}

impl Mu {
    fn new(mean: f64, scale: f64) -> BuiltinResult<Self> {
        if !mean.is_finite() || !scale.is_finite() {
            return Err(polyval_error("polyval: mu values must be finite"));
        }
        if scale == 0.0 {
            return Err(polyval_error("polyval: mu(2) must be non-zero"));
        }
        Ok(Self { mean, scale })
    }
}

#[derive(Clone)]
struct NumericArray {
    data: Vec<Complex64>,
    shape: Vec<usize>,
    all_real: bool,
}

#[derive(Clone)]
struct PolyfitStats {
    r: Matrix,
    df: f64,
    normr: f64,
    is_real: bool,
}

impl PolyfitStats {
    fn is_effective(&self) -> bool {
        self.r.len() > 0 && self.df > 0.0 && self.normr.is_finite()
    }
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl Matrix {
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    fn len(&self) -> usize {
        self.rows * self.cols
    }
}

struct ParsedOptions {
    s: Option<Value>,
    mu: Option<Value>,
}

fn is_known_empty_option(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(array) => array.data().is_empty(),
        Value::GpuTensor(handle) => handle.shape().contains(&0),
        _ => false,
    }
}

fn known_coefficient_len(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor_element_len(tensor)),
        Value::ComplexTensor(tensor) => Some(complex_tensor_element_len(tensor)),
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::GpuTensor(handle) if handle.shape().contains(&0) => Some(0),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(1),
        _ => None,
    }
}

fn stat_fields(value: &runmat_value::StructValue) -> BuiltinResult<(&Value, &Value, &Value)> {
    Ok((
        required_stat_field(value, PolyfitStatField::R)?,
        required_stat_field(value, PolyfitStatField::DegreesOfFreedom)?,
        required_stat_field(value, PolyfitStatField::ResidualNorm)?,
    ))
}

fn required_stat_field(
    value: &runmat_value::StructValue,
    identity: PolyfitStatField,
) -> BuiltinResult<&Value> {
    value.fields.get(identity.public_name()).ok_or_else(|| {
        polyval_error(format!(
            "polyval: S input is missing the field '{}'",
            identity.public_name()
        ))
    })
}

async fn preflight_host_options(
    rest: crate::call::FunctionArgumentView<'_>,
    coefficients: &Value,
) -> BuiltinResult<(Option<Mu>, Option<PolyfitStats>)> {
    // Inspect the later host control before any earlier resident statistics
    // field or numerical operand can initiate a provider readback.
    let mu = match rest.get(1) {
        Some(Value::GpuTensor(_)) | None => None,
        Some(value) => Some(parse_host_mu(value.clone())?),
    };
    let stats = match rest.first() {
        None => None,
        Some(value) if is_known_empty_option(value) => None,
        Some(Value::Struct(value)) => {
            let (r, df, normr) = stat_fields(value)?;
            let has_resident_field = [r, df, normr]
                .into_iter()
                .any(|field| matches!(field, Value::GpuTensor(_)));
            if has_resident_field {
                if let (Some(len), false) = (
                    known_coefficient_len(coefficients),
                    matches!(r, Value::GpuTensor(_)),
                ) {
                    let _ = convert_matrix((*r).clone(), len).await?;
                }
                if !matches!(df, Value::GpuTensor(_)) {
                    let _ = scalar_to_f64((*df).clone(), "polyval: S.df").await?;
                }
                if !matches!(normr, Value::GpuTensor(_)) {
                    let _ = scalar_to_f64((*normr).clone(), "polyval: S.normr").await?;
                }
                None
            } else if let Some(len) = known_coefficient_len(coefficients) {
                parse_stats(Value::Struct(value.clone()), len).await?
            } else {
                None
            }
        }
        Some(other) => {
            return Err(polyval_error(format!(
                "polyval: S input must be the structure returned by polyfit, got {other:?}"
            )))
        }
    };
    Ok((mu, stats))
}

async fn parse_option_values(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedOptions> {
    match rest.len() {
        0 => Ok(ParsedOptions { s: None, mu: None }),
        1 => Ok(ParsedOptions {
            s: if is_empty_value(&rest[0]).await? {
                None
            } else {
                Some(rest[0].clone())
            },
            mu: None,
        }),
        2 => Ok(ParsedOptions {
            s: if is_empty_value(&rest[0]).await? {
                None
            } else {
                Some(rest[0].clone())
            },
            mu: Some(rest[1].clone()),
        }),
        _ => Err(polyval_argument_error("polyval: too many input arguments")),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_coefficients(value: Value) -> BuiltinResult<(Vec<Complex64>, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered =
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
            convert_coefficients(gathered).await
        }
        Value::Tensor(tensor) => {
            ensure_vector_shape("polyval", tensor.shape())?;
            let data = tensor::tensor_values_f64(&tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect();
            Ok((data, true))
        }
        Value::ComplexTensor(mut tensor) => {
            ensure_vector_shape("polyval", tensor.shape())?;
            let all_real = complex_tensor_values_are_real(&tensor);
            let data = complex_tensor_values(&mut tensor);
            Ok((data, all_real))
        }
        Value::LogicalArray(mut array) => {
            ensure_vector_data_shape("polyval", &array.shape)?;
            let data = array
                .data
                .drain(..)
                .map(|bit| Complex64::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                .collect();
            Ok((data, true))
        }
        Value::Num(n) => Ok((vec![Complex64::new(n, 0.0)], true)),
        Value::Int(i) => Ok((vec![Complex64::new(i.to_f64(), 0.0)], true)),
        Value::Bool(flag) => Ok((
            vec![Complex64::new(if flag { 1.0 } else { 0.0 }, 0.0)],
            true,
        )),
        Value::Complex(re, im) => Ok((vec![Complex64::new(re, im)], im == 0.0)),
        other => Err(polyval_error(format!(
            "polyval: coefficients must be numeric, got {other:?}"
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_points(value: Value) -> BuiltinResult<(NumericArray, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let readback =
                runmat_accelerate_api::numeric_readback_for_handle(&handle).map_err(|source| {
                    let mut error = crate::runtime_error::with_provider_execution_source(
                        polyval_error("polyval: no exact owner for resident points"),
                        source.into(),
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?;
            let runtime = crate::context::legacy::active();
            let host = gpu_helpers::download_readback_value_in_context(runtime.as_ref(), &readback)
                .await
                .map_err(|mut error| {
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?;
            let (array, _) = convert_points(host).await?;
            Ok((array, true))
        }
        Value::Tensor(tensor) => Ok((
            NumericArray {
                data: tensor::tensor_values_f64(&tensor)
                    .into_iter()
                    .map(|re| Complex64::new(re, 0.0))
                    .collect(),
                shape: tensor.shape().to_vec(),
                all_real: true,
            },
            false,
        )),
        Value::ComplexTensor(tensor) => Ok((
            NumericArray {
                data: complex_tensor_values_ref(&tensor),
                shape: tensor.shape().to_vec(),
                all_real: complex_tensor_values_are_real(&tensor),
            },
            false,
        )),
        Value::LogicalArray(array) => Ok((
            NumericArray {
                data: array
                    .data
                    .iter()
                    .map(|&bit| Complex64::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                    .collect(),
                shape: array.shape.to_vec(),
                all_real: true,
            },
            false,
        )),
        Value::Num(n) => Ok((
            NumericArray {
                data: vec![Complex64::new(n, 0.0)],
                shape: vec![1, 1],
                all_real: true,
            },
            false,
        )),
        Value::Int(i) => Ok((
            NumericArray {
                data: vec![Complex64::new(i.to_f64(), 0.0)],
                shape: vec![1, 1],
                all_real: true,
            },
            false,
        )),
        Value::Bool(flag) => Ok((
            NumericArray {
                data: vec![Complex64::new(if flag { 1.0 } else { 0.0 }, 0.0)],
                shape: vec![1, 1],
                all_real: true,
            },
            false,
        )),
        Value::Complex(re, im) => Ok((
            NumericArray {
                data: vec![Complex64::new(re, im)],
                shape: vec![1, 1],
                all_real: im == 0.0,
            },
            false,
        )),
        other => Err(polyval_error(format!(
            "polyval: X must be numeric, got {other:?}"
        ))),
    }
}

async fn parse_mu(value: Value) -> BuiltinResult<Mu> {
    let host = match value {
        Value::GpuTensor(handle) => Value::Tensor(gpu_helpers::gather_tensor_async(&handle).await?),
        host => host,
    };
    parse_host_mu(host)
}

fn parse_host_mu(value: Value) -> BuiltinResult<Mu> {
    match value {
        Value::Tensor(tensor) => {
            if tensor_element_len(&tensor) < 2 {
                return Err(polyval_error(
                    "polyval: mu must contain at least two elements",
                ));
            }
            let values = tensor::tensor_values_f64(&tensor);
            Mu::new(values[0], values[1])
        }
        Value::LogicalArray(array) => {
            if array.data.len() < 2 {
                return Err(polyval_error(
                    "polyval: mu must contain at least two elements",
                ));
            }
            let mean = if array.data[0] != 0 { 1.0 } else { 0.0 };
            let scale = if array.data[1] != 0 { 1.0 } else { 0.0 };
            Mu::new(mean, scale)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Err(
            polyval_error("polyval: mu must be a numeric vector with at least two values"),
        ),
        Value::ComplexTensor(tensor) => {
            if complex_tensor_element_len(&tensor) < 2 {
                return Err(polyval_error(
                    "polyval: mu must contain at least two elements",
                ));
            }
            let ((mean_re, mean_im), (scale_re, scale_im)) =
                if let Some(storage) = tensor.integer_storage() {
                    let mean_re = storage
                        .real()
                        .value_at(0)
                        .expect("complex integer mu real mean")
                        .to_f64();
                    let mean_im = storage
                        .imag()
                        .value_at(0)
                        .expect("complex integer mu imag mean")
                        .to_f64();
                    let scale_re = storage
                        .real()
                        .value_at(1)
                        .expect("complex integer mu real scale")
                        .to_f64();
                    let scale_im = storage
                        .imag()
                        .value_at(1)
                        .expect("complex integer mu imag scale")
                        .to_f64();
                    ((mean_re, mean_im), (scale_re, scale_im))
                } else {
                    (tensor.materialize_f64()[0], tensor.materialize_f64()[1])
                };
            if mean_im != 0.0 || scale_im != 0.0 {
                return Err(polyval_error("polyval: mu values must be real"));
            }
            Mu::new(mean_re, scale_re)
        }
        _ => Err(polyval_error(
            "polyval: mu must be a numeric vector with at least two values",
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn parse_stats(value: Value, coeff_len: usize) -> BuiltinResult<Option<PolyfitStats>> {
    if is_empty_value(&value).await? {
        return Ok(None);
    }
    let struct_value = match value {
        Value::Struct(s) => s,
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
            return parse_stats(gathered, coeff_len).await;
        }
        other => {
            return Err(polyval_error(format!(
                "polyval: S input must be the structure returned by polyfit, got {other:?}"
            )))
        }
    };
    let (r_value, df_value, normr_value) = stat_fields(&struct_value)?;

    let (matrix, is_real) = convert_matrix((*r_value).clone(), coeff_len).await?;
    let df = scalar_to_f64((*df_value).clone(), "polyval: S.df").await?;
    let normr = scalar_to_f64((*normr_value).clone(), "polyval: S.normr").await?;

    Ok(Some(PolyfitStats {
        r: matrix,
        df,
        normr,
        is_real,
    }))
}

fn coefficient_matrix_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_matrix(value: Value, coeff_len: usize) -> BuiltinResult<(Matrix, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            convert_matrix(Value::Tensor(tensor), coeff_len).await
        }
        Value::Tensor(tensor) => {
            let (rows, cols) = coefficient_matrix_dimensions(tensor.shape());
            if rows != coeff_len || cols != coeff_len {
                return Err(polyval_error("polyval: size of S.R must match the coefficient vector"));
            }
            let data = tensor::tensor_values_f64(&tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect();
            Ok((Matrix { rows, cols, data }, true))
        }
        Value::ComplexTensor(mut tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            if rows != coeff_len || cols != coeff_len {
                return Err(polyval_error("polyval: size of S.R must match the coefficient vector"));
            }
            let imag_small = complex_tensor_values_are_real(&tensor);
            let data = complex_tensor_values(&mut tensor);
            Ok((Matrix { rows, cols, data }, imag_small))
        }
        Value::LogicalArray(array) => {
            let LogicalArray { data, shape } = array;
            let rows = shape.first().copied().unwrap_or(0);
            let cols = shape.get(1).copied().unwrap_or(0);
            if rows != coeff_len || cols != coeff_len {
                return Err(polyval_error("polyval: size of S.R must match the coefficient vector"));
            }
            let data = data
                .into_iter()
                .map(|bit| Complex64::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                .collect();
            Ok((Matrix { rows, cols, data }, true))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Err(
            polyval_error(
                "polyval: S.R must be a square numeric matrix matching the coefficient vector length",
            ),
        ),
        Value::Struct(_)
        | Value::Cell(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_) => Err(
            polyval_error(
                "polyval: S.R must be a square numeric matrix matching the coefficient vector length",
            ),
        ),
        _ => Err(
            polyval_error(
                "polyval: S.R must be a square numeric matrix matching the coefficient vector length",
            ),
        ),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn scalar_to_f64(value: Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(flag) => Ok(if flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if tensor_element_len(&tensor) != 1 {
                return Err(polyval_error(format!("{context} must be a scalar")));
            }
            Ok(tensor::tensor_value_f64(&tensor, 0))
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(polyval_error(format!("{context} must be a scalar")));
            }
            Ok(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            scalar_to_f64(Value::Tensor(tensor), context).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(polyval_error(format!("{context} must be real-valued")))
        }
        other => Err(polyval_error(format!(
            "{context} must be a scalar, got {other:?}"
        ))),
    }
}

fn apply_mu(values: &mut [Complex64], mu: Mu) -> BuiltinResult<()> {
    for v in values.iter_mut() {
        let real = normalization::standardize(v.re, mu.mean, mu.scale)
            .ok_or_else(|| polyval_error("polyval: scaled query point exceeds finite range"))?;
        *v = Complex64::new(real, v.im / mu.scale);
    }
    Ok(())
}

fn evaluate_polynomial(coeffs: &[Complex64], inputs: &[Complex64]) -> Vec<Complex64> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for &x in inputs {
        let mut acc = Complex64::new(0.0, 0.0);
        for &c in coeffs {
            acc = acc * x + c;
        }
        outputs.push(acc);
    }
    outputs
}

fn compute_prediction_interval(
    coeffs: &[Complex64],
    inputs: &[Complex64],
    stats: &PolyfitStats,
) -> BuiltinResult<Vec<f64>> {
    if !stats.is_effective() {
        return Ok(vec![0.0; inputs.len()]);
    }
    let n = coeffs.len();
    let mut delta = Vec::with_capacity(inputs.len());
    for &x in inputs {
        let row = vandermonde_row(x, n);
        let solved = solve_row_against_upper(&row, &stats.r)?;
        let sum_sq: f64 = solved.iter().map(|c| c.norm_sqr()).sum();
        let interval = (1.0 + sum_sq).sqrt() * (stats.normr / stats.df.sqrt());
        delta.push(interval);
    }
    Ok(delta)
}

fn vandermonde_row(x: Complex64, len: usize) -> Vec<Complex64> {
    if len == 0 {
        return vec![Complex64::new(1.0, 0.0)];
    }
    let degree = len - 1;
    let mut powers = vec![Complex64::new(1.0, 0.0); degree + 1];
    for idx in 1..=degree {
        powers[idx] = powers[idx - 1] * x;
    }
    let mut row = vec![Complex64::new(0.0, 0.0); degree + 1];
    for (i, value) in powers.into_iter().enumerate() {
        row[degree - i] = value;
    }
    row
}

fn solve_row_against_upper(row: &[Complex64], matrix: &Matrix) -> BuiltinResult<Vec<Complex64>> {
    let n = row.len();
    if matrix.rows != n || matrix.cols != n {
        return Err(polyval_error(
            "polyval: size of S.R must match the coefficient vector",
        ));
    }
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    // The row variance is ||v / R||². Solve Rᵀ z = vᵀ by forward
    // substitution; R is the upper-triangular factor in the same coordinate
    // system as both the published coefficients and this Vandermonde row.
    for j in 0..n {
        let mut acc = row[j];
        for (k, value) in result.iter().enumerate().take(j) {
            acc -= *value * matrix.get(k, j);
        }
        let diag = matrix.get(j, j);
        let column_scale = (0..=j)
            .map(|row| matrix.get(row, j).norm())
            .fold(0.0_f64, f64::max);
        if diag.norm() <= f64::EPSILON * column_scale {
            return Err(polyval_error("polyval: S.R is singular"));
        }
        result[j] = acc / diag;
    }
    Ok(result)
}

fn finalize_values(
    data: &[Complex64],
    shape: &[usize],
    prefer_gpu: bool,
    real_only: bool,
    residency: &resident::Request,
) -> BuiltinResult<Value> {
    if real_only {
        let real_data: Vec<f64> = data.iter().map(|c| c.re).collect();
        finalize_real(real_data, shape, prefer_gpu, residency)
    } else if data.len() == 1 {
        let value = data[0];
        Ok(Value::Complex(value.re, value.im))
    } else {
        let complex_data: Vec<(f64, f64)> = data.iter().map(|c| (c.re, c.im)).collect();
        let tensor = ComplexTensor::new(complex_data, shape.to_vec())
            .map_err(|e| polyval_error(format!("polyval: failed to build complex tensor: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn finalize_delta(
    data: Vec<f64>,
    shape: &[usize],
    prefer_gpu: bool,
    residency: &resident::Request,
) -> BuiltinResult<Value> {
    finalize_real(data, shape, prefer_gpu, residency)
}

fn finalize_real(
    data: Vec<f64>,
    shape: &[usize],
    prefer_gpu: bool,
    residency: &resident::Request,
) -> BuiltinResult<Value> {
    let tensor = Tensor::new(data, shape.to_vec())
        .map_err(|e| polyval_error(format!("polyval: failed to build tensor: {e}")))?;
    if prefer_gpu {
        if let Some(value) = residency.upload_result(&tensor)? {
            return Ok(value);
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_like(
    shape: &[usize],
    prefer_gpu: bool,
    residency: &resident::Request,
) -> BuiltinResult<Value> {
    let len = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| polyval_error("polyval: output shape overflow"))?
    };
    finalize_real(vec![0.0; len], shape, prefer_gpu, residency)
}

fn ensure_vector_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if !is_vector_shape(shape) {
        Err(polyval_error(format!(
            "{name}: coefficients must be a scalar, row vector, or column vector"
        )))
    } else {
        Ok(())
    }
}

fn ensure_vector_data_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if !is_vector_shape(shape) {
        Err(polyval_error(format!(
            "{name}: inputs must be vectors or scalars"
        )))
    } else {
        Ok(())
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|&&dim| dim > 1).count() <= 1
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_storage()
        .as_ref()
        .map_or(tensor.materialize_f64().len(), |storage| storage.len())
}

fn complex_tensor_values(tensor: &mut ComplexTensor) -> Vec<Complex64> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .real()
            .exact_values()
            .into_iter()
            .zip(storage.imag().exact_values())
            .map(|(re, im)| Complex64::new(re.to_f64(), im.to_f64()))
            .collect();
    }
    tensor
        .materialize_f64()
        .drain(..)
        .map(|(re, im)| Complex64::new(re, im))
        .collect()
}

fn complex_tensor_values_ref(tensor: &ComplexTensor) -> Vec<Complex64> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .real()
            .exact_values()
            .into_iter()
            .zip(storage.imag().exact_values())
            .map(|(re, im)| Complex64::new(re.to_f64(), im.to_f64()))
            .collect();
    }
    tensor
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect()
}

fn complex_tensor_values_are_real(tensor: &ComplexTensor) -> bool {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .imag()
            .exact_values()
            .iter()
            .all(|value| value.is_zero());
    }
    tensor.materialize_f64().iter().all(|&(_, im)| im == 0.0)
}

#[async_recursion::async_recursion(?Send)]
async fn is_empty_value(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Tensor(t) => Ok(tensor_element_len(t) == 0),
        Value::LogicalArray(l) => Ok(l.data.is_empty()),
        Value::Cell(ca) => Ok(ca.data().is_empty()),
        Value::GpuTensor(handle) => {
            let gathered =
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
            is_empty_value(&gathered).await
        }
        _ => Ok(false),
    }
}

fn values_are_real(values: &[Complex64]) -> bool {
    values.iter().all(|c| c.im == 0.0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntegerComplexStorage, IntegerStorage};

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
    }

    #[test]
    fn polyval_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = POLYVAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"y = polyval(p, x)"));
        assert!(labels.contains(&"y = polyval(p, x, S)"));
        assert!(labels.contains(&"y = polyval(p, x, S, mu)"));
        assert!(labels.contains(&"[y, delta] = polyval(p, x, S)"));
    }

    #[test]
    fn polyval_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = POLYVAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.POLYVAL.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.POLYVAL.INVALID_INPUT"));
        assert!(codes.contains(&"RM.POLYVAL.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_scalar() {
        let coeffs = Tensor::new(vec![2.0, -3.0, 5.0], vec![1, 3]).unwrap();
        let value =
            polyval_builtin(Value::Tensor(coeffs), Value::Num(4.0), Vec::new()).expect("polyval");
        match value {
            Value::Num(n) => assert!((n - (2.0 * 16.0 - 12.0 + 5.0)).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_matrix_input() {
        let coeffs = Tensor::new(vec![1.0, 0.0, -2.0, 1.0], vec![1, 4]).unwrap();
        let points = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5, 1]).unwrap();
        let value = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points.clone()),
            Vec::new(),
        )
        .expect("polyval");
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), points.shape());
                let expected = vec![-3.0, 2.0, 1.0, 0.0, 5.0];
                assert_eq!(tensor.materialize_f64(), expected);
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_complex_inputs() {
        let coeffs =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (0.0, 4.0)], vec![1, 3]).unwrap();
        let points =
            ComplexTensor::new(vec![(-1.0, 1.0), (0.0, 0.0), (1.0, -2.0)], vec![1, 3]).unwrap();
        let value = polyval_builtin(
            Value::ComplexTensor(coeffs),
            Value::ComplexTensor(points.clone()),
            Vec::new(),
        )
        .expect("polyval");
        match value {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape(), points.shape());
                assert_eq!(tensor.materialize_f64().len(), 3);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn tiny_nonzero_complex_value_and_nonzero_mu_scale_survive() {
        let tiny = polyval_builtin(Value::Complex(1.0, 1.0e-13), Value::Num(2.0), Vec::new())
            .expect("complex constant");
        assert_eq!(tiny, Value::Complex(1.0, 1.0e-13));

        let mu = Value::Tensor(Tensor::new(vec![2.0e-14, 1.0e-14], vec![1, 2]).unwrap());
        let options = vec![
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            mu,
        ];
        let scaled = polyval_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(3.0e-14),
            options,
        )
        .expect("small nonzero mu scale");
        let Value::Num(scaled) = scaled else {
            panic!("scalar prediction expected");
        };
        assert!((scaled - 3.0).abs() < 1.0e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_with_mu() {
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let mu = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let value = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::Tensor(mu),
            ],
        )
        .expect("polyval");
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.materialize_f64(), vec![0.25, 0.0, 0.25]);
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn polyval_typed_integer_coefficients_points_and_mu_cross_double_boundary_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let coeffs = Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0]), vec![1, 3]).unwrap();
        let points = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 2]), vec![1, 3]).unwrap();
        let mu = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).unwrap();
        let value = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::Tensor(mu),
            ],
        )
        .expect("polyval");
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), [1, 3]);
                assert_eq!(tensor.materialize_f64(), vec![0.25, 0.0, 0.25]);
                assert!(tensor.integer_storage().is_none());
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn polyval_complex_integer_mu_rejects_before_conversion() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![0, 0]),
        )
        .expect("complex integer mu");
        let mu = ComplexTensor::new_integer(storage, vec![1, 2]).expect("mu tensor");

        let error = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::ComplexTensor(mu),
            ],
        )
        .expect_err("typed complex integer fit options must reject");
        assert!(
            error
                .message()
                .contains("complex numbers with integer types are not supported"),
            "{error:?}"
        );
    }

    #[test]
    fn polyval_stats_fields_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        let r = Tensor::new_integer(
            IntegerStorage::I16(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]),
            vec![3, 3],
        )
        .unwrap();
        let df = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        let normr = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).unwrap();
        st.fields.insert("R".to_string(), Value::Tensor(r));
        st.fields.insert("df".to_string(), Value::Tensor(df));
        st.fields.insert("normr".to_string(), Value::Tensor(normr));

        let eval = futures::executor::block_on(evaluate(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            &[Value::Struct(st.finish())],
            true,
        ))
        .expect("polyval");

        let (_, delta) = eval.into_pair().expect("delta available");
        match delta {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), [1, 3]);
                assert!(tensor
                    .materialize_f64()
                    .iter()
                    .all(|value| value.is_finite()));
            }
            other => panic!("expected tensor delta, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_delta_computation() {
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        let r = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        st.fields.insert("R".to_string(), Value::Tensor(r));
        st.fields.insert("df".to_string(), Value::Num(4.0));
        st.fields.insert("normr".to_string(), Value::Num(2.0));
        let stats = Value::Struct(st.finish());
        let eval = futures::executor::block_on(evaluate(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            &[stats],
            true,
        ))
        .expect("polyval");
        let (_, delta) = eval.into_pair().expect("delta available");
        match delta {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), [1, 3]);
                assert_eq!(tensor.materialize_f64().len(), 3);
            }
            other => panic!("expected tensor delta, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_delta_requires_stats() {
        let coeffs = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let points = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = futures::executor::block_on(evaluate(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            &[],
            true,
        ))
        .expect_err("expected error");
        assert_error_contains(err, "S input");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_invalid_mu_length_errors() {
        let coeffs = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let points = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let mu = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let err = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![Value::Tensor(placeholder), Value::Tensor(mu)],
        )
        .expect_err("expected mu length error");
        assert_error_contains(err, "mu must contain at least two elements");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_rejects_excess_optional_arguments() {
        let coeffs = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let points = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let err = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        )
        .expect_err("expected too many arguments error");
        assert_eq!(err.identifier(), POLYVAL_ERROR_INVALID_ARGUMENT.identifier);
        assert_error_contains(err, "too many input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_complex_mu_rejected() {
        let coeffs = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let points = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let complex_mu =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.5)], vec![1, 2]).expect("complex mu");
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let err = polyval_builtin(
            Value::Tensor(coeffs),
            Value::Tensor(points),
            vec![Value::Tensor(placeholder), Value::ComplexTensor(complex_mu)],
        )
        .expect_err("expected complex mu error");
        assert_error_contains(err, "mu values must be real");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_invalid_stats_missing_r() {
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("df".to_string(), Value::Num(1.0));
        st.fields.insert("normr".to_string(), Value::Num(1.0));
        let stats = Value::Struct(st.finish());
        let err = polyval_builtin(Value::Tensor(coeffs), Value::Tensor(points), vec![stats])
            .expect_err("expected missing R error");
        assert_error_contains(err, "missing the field 'R'");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn polyval_stat_matrix_preserves_rank_zero_and_rank_one_geometry() {
        let statistics = |shape| {
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            fields.fields.insert(
                "R".to_string(),
                Value::Tensor(Tensor::new(vec![1.0], shape).expect("R tensor")),
            );
            fields.fields.insert("df".to_string(), Value::Num(1.0));
            fields.fields.insert("normr".to_string(), Value::Num(0.0));
            Value::Struct(fields.finish())
        };
        let coefficients = || Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap());

        let rank_zero_error =
            polyval_builtin(coefficients(), Value::Num(3.0), vec![statistics(vec![])])
                .expect_err("rank-zero R has zero matrix rows and columns");
        assert_error_contains(rank_zero_error, "size of S.R must match");

        let rank_one_value =
            polyval_builtin(coefficients(), Value::Num(3.0), vec![statistics(vec![1])])
                .expect("rank-one R is a 1-by-1 coefficient matrix");
        match rank_one_value {
            Value::Num(value) => assert_eq!(value, 2.0),
            other => panic!("expected scalar polynomial result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap();
            let points = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let coeff_handle = provider
                .upload(&HostTensorView {
                    data: &coeffs.materialize_f64(),
                    shape: coeffs.shape(),
                })
                .expect("upload coeff");
            let point_handle = provider
                .upload(&HostTensorView {
                    data: &points.materialize_f64(),
                    shape: points.shape(),
                })
                .expect("upload points");
            let value = polyval_builtin(
                Value::GpuTensor(coeff_handle),
                Value::GpuTensor(point_handle),
                Vec::new(),
            )
            .expect("polyval");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.materialize_f64(), vec![2.0, 1.0, 2.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn polyval_wgpu_matches_cpu_real_inputs() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![-2.0, -1.0, 0.5, 2.5], vec![4, 1]).unwrap();

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let coeff_handle = provider
            .upload(&HostTensorView {
                data: &coeffs.materialize_f64(),
                shape: coeffs.shape(),
            })
            .expect("upload coeffs");
        let point_handle = provider
            .upload(&HostTensorView {
                data: &points.materialize_f64(),
                shape: points.shape(),
            })
            .expect("upload points");

        let gpu_value = polyval_builtin(
            Value::GpuTensor(coeff_handle.clone()),
            Value::GpuTensor(point_handle.clone()),
            Vec::new(),
        )
        .expect("polyval gpu");

        drop((coeff_handle, point_handle));

        let gathered = test_support::gather(gpu_value).expect("gather");

        let coeff_complex: Vec<Complex64> = coeffs
            .materialize_f64()
            .iter()
            .map(|&c| Complex64::new(c, 0.0))
            .collect();
        let point_complex: Vec<Complex64> = points
            .materialize_f64()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        let expected_vals = evaluate_polynomial(&coeff_complex, &point_complex);
        let expected: Vec<f64> = expected_vals.iter().map(|c| c.re).collect();

        assert_eq!(gathered.shape(), [4, 1]);
        assert_eq!(gathered.materialize_f64(), expected);
    }

    fn polyval_builtin(p: Value, x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
        block_on(super::polyval_builtin(p, x, rest))
    }
}
