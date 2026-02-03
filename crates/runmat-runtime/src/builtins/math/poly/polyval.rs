//! MATLAB-compatible `polyval` builtin with GPU-aware semantics for RunMat.

use log::debug;
use num_complex::Complex64;
use runmat_accelerate_api::{HostTensorView, ProviderPolyvalMu, ProviderPolyvalOptions};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::poly::type_resolvers::polyval_type;
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;
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
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
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
    category = "math/poly",
    summary = "Evaluate a polynomial at given points with MATLAB-compatible options.",
    keywords = "polyval,polynomial,polyfit,delta,gpu",
    accel = "sink",
    sink = true,
    type_resolver(polyval_type),
    builtin_path = "crate::builtins::math::poly::polyval"
)]
async fn polyval_builtin(p: Value, x: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(p, x, &rest, false).await?;
    Ok(eval.value())
}

/// Evaluate `polyval`, optionally computing the prediction interval.
pub async fn evaluate(
    coefficients: Value,
    points: Value,
    rest: &[Value],
    want_delta: bool,
) -> BuiltinResult<PolyvalEval> {
    let options = parse_option_values(rest).await?;

    let coeff_clone = coefficients.clone();
    let points_clone = points.clone();

    let coeff_was_gpu = matches!(coefficients, Value::GpuTensor(_));
    let (coeffs, coeff_real) = convert_coefficients(coeff_clone).await?;

    let (mut inputs, prefer_gpu_points) = convert_points(points_clone).await?;
    let prefer_gpu_output = prefer_gpu_points || coeff_was_gpu;

    let mu = match options.mu.clone() {
        Some(mu_value) => Some(parse_mu(mu_value).await?),
        None => None,
    };

    if prefer_gpu_output && !want_delta && options.s.is_none() {
        if let Some(value) =
            try_gpu_polyval(&coeffs, coeff_real, &inputs, mu, prefer_gpu_output).await?
        {
            return Ok(PolyvalEval::new(value, None));
        }
    }

    if let Some(mu_val) = mu {
        apply_mu(&mut inputs.data, mu_val)?;
    }

    let stats = if let Some(s_value) = options.s {
        parse_stats(s_value, coeffs.len()).await?
    } else {
        None
    };

    if want_delta && stats.is_none() {
        return Err(polyval_error(
            "polyval: S input (structure returned by polyfit) is required for delta output",
        ));
    }

    if inputs.data.is_empty() {
        let y = zeros_like(&inputs.shape, prefer_gpu_output)?;
        let delta = if want_delta {
            Some(zeros_like(&inputs.shape, prefer_gpu_output)?)
        } else {
            None
        };
        return Ok(PolyvalEval::new(y, delta));
    }

    if coeffs.is_empty() {
        let zeros = zeros_like(&inputs.shape, prefer_gpu_output)?;
        let delta = if want_delta {
            Some(zeros_like(&inputs.shape, prefer_gpu_output)?)
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
    )?;

    let delta_value = if want_delta {
        let stats = stats.expect("delta requires stats");
        let delta = compute_prediction_interval(&coeffs, &inputs.data, &stats)?;
        let prefer = prefer_gpu_output && stats.is_real;
        Some(finalize_delta(delta, &inputs.shape, prefer)?)
    } else {
        None
    };

    Ok(PolyvalEval::new(result_value, delta_value))
}

async fn try_gpu_polyval(
    coeffs: &[Complex64],
    coeff_real: bool,
    inputs: &NumericArray,
    mu: Option<Mu>,
    prefer_gpu_output: bool,
) -> BuiltinResult<Option<Value>> {
    if !coeff_real || !inputs.all_real {
        return Ok(None);
    }
    if coeffs.is_empty() || inputs.data.is_empty() {
        return Ok(None);
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };

    let coeff_data: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
    let coeff_shape = vec![1usize, coeffs.len()];
    let coeff_view = HostTensorView {
        data: &coeff_data,
        shape: &coeff_shape,
    };
    let coeff_handle = match provider.upload(&coeff_view) {
        Ok(handle) => handle,
        Err(err) => {
            debug!("polyval: GPU upload of coefficients failed, falling back: {err}");
            return Ok(None);
        }
    };

    let input_data: Vec<f64> = inputs.data.iter().map(|c| c.re).collect();
    let input_shape = inputs.shape.clone();
    let input_view = HostTensorView {
        data: &input_data,
        shape: &input_shape,
    };
    let input_handle = match provider.upload(&input_view) {
        Ok(handle) => handle,
        Err(err) => {
            debug!("polyval: GPU upload of evaluation points failed, falling back: {err}");
            let _ = provider.free(&coeff_handle);
            return Ok(None);
        }
    };

    let options = ProviderPolyvalOptions {
        mu: mu.map(|m| ProviderPolyvalMu {
            mean: m.mean,
            scale: m.scale,
        }),
    };

    let result_handle = match provider.polyval(&coeff_handle, &input_handle, &options) {
        Ok(handle) => handle,
        Err(err) => {
            debug!("polyval: GPU kernel execution failed, falling back: {err}");
            let _ = provider.free(&coeff_handle);
            let _ = provider.free(&input_handle);
            return Ok(None);
        }
    };

    let _ = provider.free(&coeff_handle);
    let _ = provider.free(&input_handle);

    if prefer_gpu_output {
        return Ok(Some(Value::GpuTensor(result_handle)));
    }

    let host = match download_handle_async(provider, &result_handle).await {
        Ok(host) => host,
        Err(err) => {
            debug!("polyval: GPU download failed, falling back: {err}");
            let _ = provider.free(&result_handle);
            return Ok(None);
        }
    };
    let _ = provider.free(&result_handle);

    let tensor =
        Tensor::new(host.data, host.shape).map_err(|e| polyval_error(format!("polyval: {e}")))?;
    Ok(Some(tensor::tensor_into_value(tensor)))
}

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
            .ok_or_else(|| polyval_error("polyval: delta output not computed"))
    }

    /// Consume into the main value.
    pub fn into_value(self) -> Value {
        self.value
    }

    /// Consume into `(value, delta)` pair.
    pub fn into_pair(self) -> BuiltinResult<(Value, Value)> {
        match self.delta {
            Some(delta) => Ok((self.value, delta)),
            None => Err(polyval_error("polyval: delta output not computed")),
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
        if scale.abs() <= EPS {
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

async fn parse_option_values(rest: &[Value]) -> BuiltinResult<ParsedOptions> {
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
        _ => Err(polyval_error("polyval: too many input arguments")),
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
        Value::Tensor(mut tensor) => {
            ensure_vector_shape("polyval", &tensor.shape)?;
            let data = tensor
                .data
                .drain(..)
                .map(|re| Complex64::new(re, 0.0))
                .collect();
            Ok((data, true))
        }
        Value::ComplexTensor(mut tensor) => {
            ensure_vector_shape("polyval", &tensor.shape)?;
            let all_real = tensor.data.iter().all(|&(_, im)| im.abs() <= EPS);
            let data = tensor
                .data
                .drain(..)
                .map(|(re, im)| Complex64::new(re, im))
                .collect();
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
        Value::Complex(re, im) => Ok((vec![Complex64::new(re, im)], im.abs() <= EPS)),
        other => Err(polyval_error(format!(
            "polyval: coefficients must be numeric, got {other:?}"
        ))),
    }
}

async fn convert_points(value: Value) -> BuiltinResult<(NumericArray, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let array = NumericArray {
                data: tensor
                    .data
                    .iter()
                    .map(|&re| Complex64::new(re, 0.0))
                    .collect(),
                shape: tensor.shape.clone(),
                all_real: true,
            };
            Ok((array, true))
        }
        Value::Tensor(tensor) => Ok((
            NumericArray {
                data: tensor
                    .data
                    .iter()
                    .map(|&re| Complex64::new(re, 0.0))
                    .collect(),
                shape: tensor.shape.clone(),
                all_real: true,
            },
            false,
        )),
        Value::ComplexTensor(tensor) => Ok((
            NumericArray {
                data: tensor
                    .data
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect(),
                shape: tensor.shape.clone(),
                all_real: tensor.data.iter().all(|&(_, im)| im.abs() <= EPS),
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
                shape: array.shape.clone(),
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
                all_real: im.abs() <= EPS,
            },
            false,
        )),
        other => Err(polyval_error(format!(
            "polyval: X must be numeric, got {other:?}"
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn parse_mu(value: Value) -> BuiltinResult<Mu> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
            parse_mu(Value::Tensor(gathered)).await
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() < 2 {
                return Err(polyval_error(
                    "polyval: mu must contain at least two elements",
                ));
            }
            Mu::new(tensor.data[0], tensor.data[1])
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
            if tensor.data.len() < 2 {
                return Err(polyval_error(
                    "polyval: mu must contain at least two elements",
                ));
            }
            let (mean_re, mean_im) = tensor.data[0];
            let (scale_re, scale_im) = tensor.data[1];
            if mean_im.abs() > EPS || scale_im.abs() > EPS {
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
    let r_value = struct_value
        .fields
        .get("R")
        .cloned()
        .ok_or_else(|| polyval_error("polyval: S input is missing the field 'R'"))?;
    let df_value = struct_value
        .fields
        .get("df")
        .cloned()
        .ok_or_else(|| polyval_error("polyval: S input is missing the field 'df'"))?;
    let normr_value = struct_value
        .fields
        .get("normr")
        .cloned()
        .ok_or_else(|| polyval_error("polyval: S input is missing the field 'normr'"))?;

    let (matrix, is_real) = convert_matrix(r_value, coeff_len).await?;
    let df = scalar_to_f64(df_value, "polyval: S.df").await?;
    let normr = scalar_to_f64(normr_value, "polyval: S.normr").await?;

    Ok(Some(PolyfitStats {
        r: matrix,
        df,
        normr,
        is_real,
    }))
}

#[async_recursion::async_recursion(?Send)]
async fn convert_matrix(value: Value, coeff_len: usize) -> BuiltinResult<(Matrix, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            convert_matrix(Value::Tensor(tensor), coeff_len).await
        }
        Value::Tensor(tensor) => {
            let Tensor {
                data, rows, cols, ..
            } = tensor;
            if rows != coeff_len || cols != coeff_len {
                return Err(polyval_error("polyval: size of S.R must match the coefficient vector"));
            }
            let data = data.into_iter().map(|re| Complex64::new(re, 0.0)).collect();
            Ok((Matrix { rows, cols, data }, true))
        }
        Value::ComplexTensor(tensor) => {
            let ComplexTensor {
                data, rows, cols, ..
            } = tensor;
            if rows != coeff_len || cols != coeff_len {
                return Err(polyval_error("polyval: size of S.R must match the coefficient vector"));
            }
            let imag_small = data.iter().all(|&(_, im)| im.abs() <= EPS);
            let data = data
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im))
                .collect();
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
            if tensor.data.len() != 1 {
                return Err(polyval_error(format!("{context} must be a scalar")));
            }
            Ok(tensor.data[0])
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
    let mean = Complex64::new(mu.mean, 0.0);
    let scale = Complex64::new(mu.scale, 0.0);
    for v in values.iter_mut() {
        *v = (*v - mean) / scale;
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
    for j in (0..n).rev() {
        let mut acc = row[j];
        for (k, value) in result.iter().enumerate().skip(j + 1) {
            acc -= *value * matrix.get(k, j);
        }
        let diag = matrix.get(j, j);
        if diag.norm() <= EPS {
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
) -> BuiltinResult<Value> {
    if real_only {
        let real_data: Vec<f64> = data.iter().map(|c| c.re).collect();
        finalize_real(real_data, shape, prefer_gpu)
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

fn finalize_delta(data: Vec<f64>, shape: &[usize], prefer_gpu: bool) -> BuiltinResult<Value> {
    finalize_real(data, shape, prefer_gpu)
}

fn finalize_real(data: Vec<f64>, shape: &[usize], prefer_gpu: bool) -> BuiltinResult<Value> {
    let tensor = Tensor::new(data, shape.to_vec())
        .map_err(|e| polyval_error(format!("polyval: failed to build tensor: {e}")))?;
    if prefer_gpu {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_like(shape: &[usize], prefer_gpu: bool) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    finalize_real(vec![0.0; len], shape, prefer_gpu)
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

#[async_recursion::async_recursion(?Send)]
async fn is_empty_value(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Tensor(t) => Ok(t.data.is_empty()),
        Value::LogicalArray(l) => Ok(l.data.is_empty()),
        Value::Cell(ca) => Ok(ca.data.is_empty()),
        Value::GpuTensor(handle) => {
            let gathered =
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
            is_empty_value(&gathered).await
        }
        _ => Ok(false),
    }
}

fn values_are_real(values: &[Complex64]) -> bool {
    values.iter().all(|c| c.im.abs() <= EPS)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::StructValue;

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
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
                assert_eq!(tensor.shape, points.shape);
                let expected = vec![-3.0, 2.0, 1.0, 0.0, 5.0];
                assert_eq!(tensor.data, expected);
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
                assert_eq!(tensor.shape, points.shape);
                assert_eq!(tensor.data.len(), 3);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
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
                assert_eq!(tensor.data, vec![0.25, 0.0, 0.25]);
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_delta_computation() {
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![1, 3]).unwrap();
        let points = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let mut st = StructValue::new();
        let r = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        st.fields.insert("R".to_string(), Value::Tensor(r));
        st.fields.insert("df".to_string(), Value::Num(4.0));
        st.fields.insert("normr".to_string(), Value::Num(2.0));
        let stats = Value::Struct(st);
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
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data.len(), 3);
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
        let mut st = StructValue::new();
        st.fields.insert("df".to_string(), Value::Num(1.0));
        st.fields.insert("normr".to_string(), Value::Num(1.0));
        let stats = Value::Struct(st);
        let err = polyval_builtin(Value::Tensor(coeffs), Value::Tensor(points), vec![stats])
            .expect_err("expected missing R error");
        assert_error_contains(err, "missing the field 'R'");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyval_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap();
            let points = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let coeff_handle = provider
                .upload(&HostTensorView {
                    data: &coeffs.data,
                    shape: &coeffs.shape,
                })
                .expect("upload coeff");
            let point_handle = provider
                .upload(&HostTensorView {
                    data: &points.data,
                    shape: &points.shape,
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
                    assert_eq!(gathered.data, vec![2.0, 1.0, 2.0]);
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
                data: &coeffs.data,
                shape: &coeffs.shape,
            })
            .expect("upload coeffs");
        let point_handle = provider
            .upload(&HostTensorView {
                data: &points.data,
                shape: &points.shape,
            })
            .expect("upload points");

        let gpu_value = polyval_builtin(
            Value::GpuTensor(coeff_handle.clone()),
            Value::GpuTensor(point_handle.clone()),
            Vec::new(),
        )
        .expect("polyval gpu");

        let _ = provider.free(&coeff_handle);
        let _ = provider.free(&point_handle);

        let gathered = test_support::gather(gpu_value).expect("gather");

        let coeff_complex: Vec<Complex64> = coeffs
            .data
            .iter()
            .map(|&c| Complex64::new(c, 0.0))
            .collect();
        let point_complex: Vec<Complex64> = points
            .data
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        let expected_vals = evaluate_polynomial(&coeff_complex, &point_complex);
        let expected: Vec<f64> = expected_vals.iter().map(|c| c.re).collect();

        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.data, expected);
    }

    fn polyval_builtin(p: Value, x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::polyval_builtin(p, x, rest))
    }
}
