//! MATLAB-compatible `polyfit` builtin with GPU-aware semantics for RunMat.

use log::{trace, warn};
use num_complex::Complex64;
use runmat_accelerate_api::ProviderPolyfitResult;
use runmat_builtins::{ComplexTensor, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::dispatcher;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::poly::type_resolvers::polyfit_type;

const EPS: f64 = 1.0e-12;
const EPS_NAN: f64 = 1.0e-12;
const BUILTIN_NAME: &str = "polyfit";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::polyfit")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polyfit",
    op_kind: GpuOpKind::Custom("polyfit"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("polyfit")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may gather to the host and invoke the shared Householder QR solver; WGPU implements this path today.",
};

fn polyfit_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::polyfit")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polyfit",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a sink node—polynomial fitting materialises results eagerly and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "polyfit",
    category = "math/poly",
    summary = "Fit an n-th degree polynomial to data points with MATLAB-compatible outputs.",
    keywords = "polyfit,polynomial,least-squares,gpu",
    accel = "sink",
    sink = true,
    type_resolver(polyfit_type),
    builtin_path = "crate::builtins::math::poly::polyfit"
)]
async fn polyfit_builtin(
    x: Value,
    y: Value,
    degree: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate(x, y, degree, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut outputs = vec![eval.coefficients()];
        if out_count >= 2 {
            outputs.push(eval.stats());
        }
        if out_count >= 3 {
            outputs.push(eval.mu());
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            outputs,
        ));
    }
    Ok(eval.coefficients())
}

/// Evaluate `polyfit`, returning the multi-output envelope used by the VM.
pub async fn evaluate(
    x: Value,
    y: Value,
    degree: Value,
    rest: &[Value],
) -> BuiltinResult<PolyfitEval> {
    let deg = parse_degree(&degree)?;

    if let Some(eval) = try_gpu_polyfit(&x, &y, deg, rest).await? {
        return Ok(eval);
    }

    let x_host = dispatcher::gather_if_needed_async(&x).await?;
    let y_host = dispatcher::gather_if_needed_async(&y).await?;

    let x_data = real_vector("polyfit", "X", x_host).await?;
    let (y_data, is_complex_input) = complex_vector("polyfit", "Y", y_host).await?;

    if x_data.len() != y_data.len() {
        return Err(polyfit_error(
            "polyfit: X and Y vectors must be the same length",
        ));
    }
    if x_data.is_empty() {
        return Err(polyfit_error(
            "polyfit: X and Y must contain at least one sample",
        ));
    }
    if deg + 1 > x_data.len() && x_data.len() > 1 {
        warn!(
            "polyfit: polynomial degree {} is ill-conditioned for {} data points; results may be inaccurate",
            deg,
            x_data.len()
        );
    }

    let weights = parse_weights(rest, x_data.len()).await?;
    let mut solution = solve_polyfit(&x_data, &y_data, deg, weights.as_deref())?;
    if is_complex_input {
        solution.is_complex = true;
    }

    PolyfitEval::from_solution(solution)
}

async fn try_gpu_polyfit(
    x: &Value,
    y: &Value,
    degree: usize,
    rest: &[Value],
) -> BuiltinResult<Option<PolyfitEval>> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let x_handle = match x {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };
    let y_handle = match y {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    if rest.len() > 1 {
        return Ok(None);
    }

    let weight_handle = match rest.first() {
        Some(Value::GpuTensor(handle)) => Some(handle),
        Some(_) => return Ok(None),
        None => None,
    };

    let result = match provider
        .polyfit(x_handle, y_handle, degree, weight_handle)
        .await
    {
        Ok(res) => res,
        Err(err) => {
            trace!("polyfit: provider path unavailable ({err}); falling back to host");
            return Ok(None);
        }
    };

    let solution = PolyfitSolution::from_provider(result)?;
    PolyfitEval::from_solution(solution).map(Some)
}

#[derive(Clone, Debug)]
struct PolyfitSolution {
    coeffs: Vec<Complex64>,
    r_matrix: Vec<f64>,
    mu_mean: f64,
    mu_scale: f64,
    normr: f64,
    df: f64,
    cols: usize,
    is_complex: bool,
}

impl PolyfitSolution {
    fn from_provider(result: ProviderPolyfitResult) -> BuiltinResult<Self> {
        let cols = result.coefficients.len();
        if cols == 0 {
            return Err(polyfit_error(
                "polyfit: provider returned empty coefficient vector",
            ));
        }
        if result.r_matrix.len() != cols * cols {
            return Err(polyfit_error(
                "polyfit: provider returned malformed R matrix",
            ));
        }
        let [mu_mean, mu_scale] = result.mu;
        Ok(Self {
            coeffs: result
                .coefficients
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect(),
            r_matrix: result.r_matrix,
            mu_mean,
            mu_scale,
            normr: result.normr,
            df: result.df,
            cols,
            is_complex: false,
        })
    }
}

/// Multi-output envelope for `polyfit`, mirroring MATLAB semantics.
#[derive(Debug)]
pub struct PolyfitEval {
    coefficients: Value,
    stats: Value,
    mu: Value,
    is_complex: bool,
}

impl PolyfitEval {
    fn from_solution(solution: PolyfitSolution) -> BuiltinResult<Self> {
        let coefficients = coefficients_to_value(&solution.coeffs)?;
        let stats = build_stats(
            &solution.r_matrix,
            solution.cols,
            solution.normr,
            solution.df,
        )?;
        let mu = build_mu(solution.mu_mean, solution.mu_scale)?;
        Ok(Self {
            coefficients,
            stats,
            mu,
            is_complex: solution.is_complex,
        })
    }

    /// Polynomial coefficients ordered from highest power to constant term.
    pub fn coefficients(&self) -> Value {
        self.coefficients.clone()
    }

    /// Structure `S` containing fields `R`, `df`, and `normr`.
    pub fn stats(&self) -> Value {
        self.stats.clone()
    }

    /// Centering and scaling vector `[mean(x), std(x)]`.
    pub fn mu(&self) -> Value {
        self.mu.clone()
    }

    /// Returns `true` if the fitted polynomial contains a complex coefficient.
    pub fn is_complex(&self) -> bool {
        self.is_complex
    }
}

fn parse_degree(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(polyfit_error(
                    "polyfit: degree must be a non-negative integer",
                ));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(polyfit_error("polyfit: degree must be finite"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > EPS {
                return Err(polyfit_error("polyfit: degree must be an integer"));
            }
            if rounded < 0.0 {
                return Err(polyfit_error(
                    "polyfit: degree must be a non-negative integer",
                ));
            }
            Ok(rounded as usize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => parse_degree(&Value::Num(t.data[0])),
        Value::LogicalArray(l) if l.len() == 1 => {
            parse_degree(&Value::Num(if l.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        other => Err(polyfit_error(format!(
            "polyfit: degree must be a scalar numeric value, got {other:?}"
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn real_vector(context: &str, label: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(mut tensor) => {
            ensure_vector_shape(context, label, &tensor.shape)?;
            Ok(tensor.data.drain(..).collect())
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(polyfit_error)?;
            ensure_vector_shape(context, label, &tensor.shape)?;
            Ok(tensor.data)
        }
        Value::Num(n) => Ok(vec![n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }]),
        Value::GpuTensor(handle) => {
            let gathered =
                crate::builtins::common::gpu_helpers::gather_tensor_async(&handle).await?;
            real_vector(context, label, Value::Tensor(gathered)).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(polyfit_error(format!(
            "{context}: {label} must be real-valued; complex inputs are not supported"
        ))),
        other => Err(polyfit_error(format!(
            "{context}: expected {label} to be a numeric vector, got {other:?}"
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn complex_vector(
    context: &str,
    label: &str,
    value: Value,
) -> BuiltinResult<(Vec<Complex64>, bool)> {
    match value {
        Value::Tensor(mut tensor) => {
            ensure_vector_shape(context, label, &tensor.shape)?;
            let all_real = true;
            let data = tensor
                .data
                .drain(..)
                .map(|x| Complex64::new(x, 0.0))
                .collect();
            Ok((data, all_real))
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(context, label, &tensor.shape)?;
            let is_complex = tensor.data.iter().any(|&(_, im)| im.abs() > EPS);
            let data = tensor
                .data
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>();
            Ok((data, is_complex))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(polyfit_error)?;
            ensure_vector_shape(context, label, &tensor.shape)?;
            Ok((
                tensor
                    .data
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect(),
                false,
            ))
        }
        Value::Num(n) => Ok((vec![Complex64::new(n, 0.0)], false)),
        Value::Int(i) => Ok((vec![Complex64::new(i.to_f64(), 0.0)], false)),
        Value::Bool(b) => Ok((vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)], false)),
        Value::Complex(re, im) => Ok((vec![Complex64::new(re, im)], im.abs() > EPS)),
        Value::GpuTensor(handle) => {
            let gathered =
                crate::builtins::common::gpu_helpers::gather_tensor_async(&handle).await?;
            complex_vector(context, label, Value::Tensor(gathered)).await
        }
        other => Err(polyfit_error(format!(
            "{context}: expected {label} to be a numeric vector, got {other:?}"
        ))),
    }
}

async fn parse_weights(rest: &[Value], len: usize) -> BuiltinResult<Option<Vec<f64>>> {
    match rest.len() {
        0 => Ok(None),
        1 => {
            let gathered = dispatcher::gather_if_needed_async(&rest[0]).await?;
            let data = real_vector("polyfit", "weights", gathered).await?;
            if data.len() != len {
                return Err(polyfit_error(
                    "polyfit: weight vector must match the size of X",
                ));
            }
            validate_weights(&data)?;
            Ok(Some(data))
        }
        _ => Err(polyfit_error("polyfit: too many input arguments")),
    }
}

fn validate_weights(weights: &[f64]) -> BuiltinResult<()> {
    for (idx, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(polyfit_error(format!(
                "polyfit: weight at position {} must be finite",
                idx + 1
            )));
        }
        if *w < 0.0 {
            return Err(polyfit_error("polyfit: weights must be non-negative"));
        }
    }
    Ok(())
}

fn solve_polyfit(
    x_data: &[f64],
    y_data: &[Complex64],
    degree: usize,
    weights: Option<&[f64]>,
) -> BuiltinResult<PolyfitSolution> {
    if x_data.len() != y_data.len() {
        return Err(polyfit_error(
            "polyfit: X and Y vectors must be the same length",
        ));
    }
    if x_data.is_empty() {
        return Err(polyfit_error(
            "polyfit: X and Y must contain at least one sample",
        ));
    }
    if let Some(w) = weights {
        if w.len() != x_data.len() {
            return Err(polyfit_error(
                "polyfit: weight vector must match the size of X",
            ));
        }
        validate_weights(w)?;
    }

    let mean = x_data.iter().sum::<f64>() / x_data.len() as f64;
    if !mean.is_finite() {
        return Err(polyfit_error("polyfit: mean of X must be finite"));
    }
    let scale = compute_scale(x_data, mean)?;
    let scaled: Vec<f64> = x_data.iter().map(|&v| (v - mean) / scale).collect();

    let mut rhs = y_data.to_vec();
    for (idx, value) in rhs.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(polyfit_error(format!(
                "polyfit: Y must contain finite values (encountered NaN/Inf at position {})",
                idx + 1
            )));
        }
    }
    if let Some(w) = weights {
        apply_weights_rhs(&mut rhs, w)?;
    }

    let rows = scaled.len();
    let cols = degree + 1;
    let mut vandermonde = build_vandermonde(&scaled, cols);
    if let Some(w) = weights {
        apply_weights_matrix(&mut vandermonde, rows, cols, w)?;
    }

    let mut transformed_rhs = rhs.clone();
    householder_qr(&mut vandermonde, rows, cols, &mut transformed_rhs)?;
    let coeff_scaled = solve_upper(&vandermonde, rows, cols, &transformed_rhs)?;
    let coeff_original = transform_coefficients(&coeff_scaled, mean, scale);

    let normr = residual_norm(&transformed_rhs, rows, cols);
    let df = if rows > cols {
        (rows - cols) as f64
    } else {
        0.0
    };
    let r_matrix = extract_upper(&vandermonde, rows, cols);
    let is_complex = coeff_original.iter().any(|c| c.im.abs() > EPS_NAN);

    Ok(PolyfitSolution {
        coeffs: coeff_original,
        r_matrix,
        mu_mean: mean,
        mu_scale: scale,
        normr,
        df,
        cols,
        is_complex,
    })
}

fn compute_scale(data: &[f64], mean: f64) -> BuiltinResult<f64> {
    if data.len() <= 1 {
        return Ok(1.0);
    }
    let mut acc = 0.0;
    for &value in data {
        if !value.is_finite() {
            return Err(polyfit_error("polyfit: X must contain finite values"));
        }
        let diff = value - mean;
        acc += diff * diff;
    }
    let denom = (data.len() as f64 - 1.0).max(1.0);
    let std = (acc / denom).sqrt();
    let scale = if std.abs() <= EPS { 1.0 } else { std };
    if !scale.is_finite() {
        return Err(polyfit_error(
            "polyfit: failed to compute a stable scaling factor",
        ));
    }
    Ok(scale)
}

fn build_vandermonde(u: &[f64], cols: usize) -> Vec<f64> {
    let rows = u.len();
    let mut matrix = vec![0.0; rows * cols];
    if cols == 0 {
        return matrix;
    }
    for (row_idx, &value) in u.iter().enumerate() {
        let mut powers = vec![0.0; cols];
        powers[cols - 1] = 1.0;
        for idx in (0..cols - 1).rev() {
            powers[idx] = powers[idx + 1] * value;
        }
        for col_idx in 0..cols {
            matrix[row_idx + col_idx * rows] = powers[col_idx];
        }
    }
    matrix
}

fn apply_weights_matrix(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
) -> BuiltinResult<()> {
    for (row, weight) in weights.iter().enumerate().take(rows) {
        let sqrt_w = weight.sqrt();
        if !sqrt_w.is_finite() {
            return Err(polyfit_error(format!(
                "polyfit: weight at position {} must be finite",
                row + 1
            )));
        }
        for col in 0..cols {
            let idx = row + col * rows;
            matrix[idx] *= sqrt_w;
        }
    }
    Ok(())
}

fn apply_weights_rhs(rhs: &mut [Complex64], weights: &[f64]) -> BuiltinResult<()> {
    for (idx, (value, weight)) in rhs.iter_mut().zip(weights.iter()).enumerate() {
        let sqrt_w = weight.sqrt();
        if !sqrt_w.is_finite() {
            return Err(polyfit_error(format!(
                "polyfit: weight at position {} must be finite",
                idx + 1
            )));
        }
        *value *= sqrt_w;
    }
    Ok(())
}

fn ensure_vector_shape(context: &str, label: &str, shape: &[usize]) -> BuiltinResult<()> {
    if !is_vector_shape(shape) {
        return Err(polyfit_error(format!(
            "{context}: {label} must be a vector"
        )));
    }
    Ok(())
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

fn householder_qr(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    rhs: &mut [Complex64],
) -> BuiltinResult<()> {
    let min_dim = rows.min(cols);
    for k in 0..min_dim {
        let mut norm_sq = 0.0;
        for row in k..rows {
            let val = matrix[row + k * rows];
            norm_sq += val * val;
        }
        if norm_sq <= EPS {
            continue;
        }
        let norm = norm_sq.sqrt();
        let x0 = matrix[k + k * rows];
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; rows - k];
        v[0] = x0 - alpha;
        for row in (k + 1)..rows {
            v[row - k] = matrix[row + k * rows];
        }
        let v_norm_sq: f64 = v.iter().map(|&x| x * x).sum();
        if v_norm_sq <= EPS {
            continue;
        }
        let beta = 2.0 / v_norm_sq;
        matrix[k + k * rows] = alpha;
        for row in (k + 1)..rows {
            matrix[row + k * rows] = 0.0;
        }
        for col in (k + 1)..cols {
            let mut dot = 0.0;
            for (idx, &vi) in v.iter().enumerate() {
                let row_idx = k + idx;
                dot += vi * matrix[row_idx + col * rows];
            }
            let factor = beta * dot;
            for (idx, &vi) in v.iter().enumerate() {
                let row_idx = k + idx;
                matrix[row_idx + col * rows] -= factor * vi;
            }
        }
        let mut dot = Complex64::new(0.0, 0.0);
        for (idx, &vi) in v.iter().enumerate() {
            let row_idx = k + idx;
            dot += rhs[row_idx] * vi;
        }
        let factor = Complex64::new(beta, 0.0) * dot;
        for (idx, &vi) in v.iter().enumerate() {
            let row_idx = k + idx;
            rhs[row_idx] -= factor * vi;
        }
    }
    Ok(())
}

fn solve_upper(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    rhs: &[Complex64],
) -> BuiltinResult<Vec<Complex64>> {
    if rhs.len() < rows {
        return Err(polyfit_error(
            "polyfit internal error: RHS dimension mismatch",
        ));
    }
    let mut coeffs = vec![Complex64::new(0.0, 0.0); cols];
    for col in (0..cols).rev() {
        let diag = if col < rows {
            matrix[col + col * rows]
        } else {
            0.0
        };
        if diag.abs() <= EPS {
            coeffs[col] = Complex64::new(0.0, 0.0);
            continue;
        }
        let mut acc = if col < rows {
            rhs[col]
        } else {
            Complex64::new(0.0, 0.0)
        };
        for next in (col + 1)..cols {
            let idx = if col < rows {
                matrix[col + next * rows]
            } else {
                0.0
            };
            acc -= Complex64::new(idx, 0.0) * coeffs[next];
        }
        coeffs[col] = acc / Complex64::new(diag, 0.0);
    }
    Ok(coeffs)
}

fn residual_norm(rhs: &[Complex64], rows: usize, cols: usize) -> f64 {
    let tail_start = rows.min(cols);
    let mut acc = 0.0;
    for value in rhs.iter().skip(tail_start) {
        acc += value.norm_sqr();
    }
    acc.sqrt()
}

fn extract_upper(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut output = vec![0.0; cols * cols];
    for col in 0..cols {
        for row in 0..=col {
            if row < rows {
                output[row + col * cols] = matrix[row + col * rows];
            }
        }
    }
    output
}

fn transform_coefficients(coeffs: &[Complex64], mean: f64, scale: f64) -> Vec<Complex64> {
    let mut poly: Vec<Complex64> = Vec::new();
    for &coeff in coeffs {
        let mut next = vec![Complex64::new(0.0, 0.0); poly.len() + 1];
        for (idx, &value) in poly.iter().enumerate() {
            next[idx + 1] += value / scale;
            next[idx] -= value * (mean / scale);
        }
        next[0] += coeff;
        poly = next;
    }
    poly.reverse();
    poly
}

fn coefficients_to_value(coeffs: &[Complex64]) -> BuiltinResult<Value> {
    let all_real = coeffs
        .iter()
        .all(|c| c.im.abs() <= EPS_NAN && c.re.is_finite());
    if all_real {
        let data: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
        let tensor = Tensor::new(data, vec![1, coeffs.len()])
            .map_err(|e| polyfit_error(format!("polyfit: {e}")))?;
        Ok(Value::Tensor(tensor))
    } else {
        let data: Vec<(f64, f64)> = coeffs.iter().map(|c| (c.re, c.im)).collect();
        let tensor = ComplexTensor::new(data, vec![1, coeffs.len()])
            .map_err(|e| polyfit_error(format!("polyfit: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn build_stats(r: &[f64], n: usize, normr: f64, df: f64) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(r.to_vec(), vec![n, n]).map_err(|e| polyfit_error(format!("polyfit: {e}")))?;
    let mut st = StructValue::new();
    st.fields.insert("R".to_string(), Value::Tensor(tensor));
    st.fields.insert("df".to_string(), Value::Num(df));
    st.fields.insert("normr".to_string(), Value::Num(normr));
    Ok(Value::Struct(st))
}

fn build_mu(mean: f64, scale: f64) -> BuiltinResult<Value> {
    if !scale.is_finite() || scale.abs() <= EPS {
        return Err(polyfit_error("polyfit: mu(2) must be non-zero and finite"));
    }
    let tensor = Tensor::new(vec![mean, scale], vec![1, 2])
        .map_err(|e| polyfit_error(format!("polyfit: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[derive(Debug, Clone)]
pub struct PolyfitHostRealResult {
    pub coefficients: Vec<f64>,
    pub r_matrix: Vec<f64>,
    pub mu: [f64; 2],
    pub normr: f64,
    pub df: f64,
}

pub fn polyfit_host_real_for_provider(
    x: &[f64],
    y: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
) -> BuiltinResult<PolyfitHostRealResult> {
    if x.len() != y.len() {
        return Err(polyfit_error(
            "polyfit: X and Y vectors must be the same length",
        ));
    }
    if let Some(w) = weights {
        if w.len() != x.len() {
            return Err(polyfit_error(
                "polyfit: weight vector must match the size of X",
            ));
        }
        validate_weights(w)?;
    }
    let complex_y: Vec<Complex64> = y.iter().copied().map(|v| Complex64::new(v, 0.0)).collect();
    let solution = solve_polyfit(x, &complex_y, degree, weights)?;
    let PolyfitSolution {
        coeffs,
        r_matrix,
        mu_mean,
        mu_scale,
        normr,
        df,
        cols: _,
        is_complex,
    } = solution;
    if is_complex {
        return Err(polyfit_error(
            "polyfit: provider fallback produced complex coefficients for real data",
        ));
    }
    let coeffs: Vec<f64> = coeffs.into_iter().map(|c| c.re).collect();
    let mu = [mu_mean, mu_scale];
    Ok(PolyfitHostRealResult {
        coefficients: coeffs,
        r_matrix,
        mu,
        normr,
        df,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
    }

    fn evaluate(
        x: Value,
        y: Value,
        degree: Value,
        rest: &[Value],
    ) -> Result<PolyfitEval, RuntimeError> {
        block_on(super::evaluate(x, y, degree, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fits_linear_data() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let mut y_vals = Vec::new();
        for i in 0..4 {
            y_vals.push(1.5 * i as f64 + 2.0);
        }
        let y = Tensor::new(y_vals, vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            &[],
        )
        .expect("polyfit");
        match eval.coefficients() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - 1.5).abs() < 1e-10);
                assert!((t.data[1] - 2.0).abs() < 1e-10);
            }
            other => panic!("expected tensor coefficients, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn returns_struct_and_mu() {
        let x = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[],
        )
        .expect("polyfit");
        match eval.stats() {
            Value::Struct(s) => {
                assert!(s.fields.contains_key("R"));
                assert!(s.fields.contains_key("df"));
                assert!(s.fields.contains_key("normr"));
            }
            other => panic!("expected struct, got {other:?}"),
        }
        match eval.mu() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0]).abs() < 1e-10);
                assert!(t.data[1].abs() > 0.0);
            }
            other => panic!("expected tensor mu, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weighted_fit_matches_unweighted_when_weights_equal() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let eval_unweighted = evaluate(
            Value::Tensor(x.clone()),
            Value::Tensor(y.clone()),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[],
        )
        .expect("polyfit");
        let eval_weighted = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[Value::Tensor(weights)],
        )
        .expect("polyfit");
        assert_eq!(eval_unweighted.coefficients(), eval_weighted.coefficients());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_logical_degree_scalar() {
        let x = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let logical = runmat_builtins::LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::LogicalArray(logical),
            &[],
        )
        .expect("polyfit");
        assert!(matches!(eval.coefficients(), Value::Tensor(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_integer_degree() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let err = evaluate(Value::Tensor(x), Value::Tensor(y), Value::Num(1.5), &[])
            .expect_err("polyfit should reject non-integer degree");
        assert_error_contains(err, "integer");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_infinite_weights() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, f64::INFINITY, 1.0], vec![3, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[Value::Tensor(weights)],
        )
        .expect_err("polyfit should reject infinite weights");
        assert_error_contains(err, "weight at position 2");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_inputs_are_gathered() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let x_handle = provider.upload(&view).expect("upload");
            let view_y = runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            };
            let y_handle = provider.upload(&view_y).expect("upload");
            let eval = evaluate(
                Value::GpuTensor(x_handle),
                Value::GpuTensor(y_handle),
                Value::Int(runmat_builtins::IntValue::I32(2)),
                &[],
            )
            .expect("polyfit");
            assert!(matches!(eval.coefficients(), Value::Tensor(_)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_weights_are_gathered() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
            let weights = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();

            let x_view = runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let y_view = runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            };
            let w_view = runmat_accelerate_api::HostTensorView {
                data: &weights.data,
                shape: &weights.shape,
            };

            let x_handle = provider.upload(&x_view).expect("upload x");
            let y_handle = provider.upload(&y_view).expect("upload y");
            let w_handle = provider.upload(&w_view).expect("upload weights");

            let cpu_eval = evaluate(
                Value::Tensor(x.clone()),
                Value::Tensor(y.clone()),
                Value::Int(runmat_builtins::IntValue::I32(2)),
                &[Value::Tensor(weights.clone())],
            )
            .expect("cpu polyfit");

            let gpu_eval = evaluate(
                Value::GpuTensor(x_handle.clone()),
                Value::GpuTensor(y_handle.clone()),
                Value::Int(runmat_builtins::IntValue::I32(2)),
                &[Value::GpuTensor(w_handle.clone())],
            )
            .expect("gpu polyfit with weights");

            assert_eq!(cpu_eval.coefficients(), gpu_eval.coefficients());
            assert_eq!(cpu_eval.mu(), gpu_eval.mu());

            let _ = provider.free(&x_handle);
            let _ = provider.free(&y_handle);
            let _ = provider.free(&w_handle);
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyfit_wgpu_matches_cpu() {
        let options = runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default();
        let _provider =
            match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(options) {
                Ok(p) => p,
                Err(err) => {
                    warn!("polyfit_wgpu_matches_cpu: skipping test ({err})");
                    return;
                }
            };
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0, 13.0], vec![4, 1]).unwrap();

        let cpu_eval = evaluate(
            Value::Tensor(x.clone()),
            Value::Tensor(y.clone()),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[],
        )
        .expect("cpu polyfit");

        let trait_provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let x_view = runmat_accelerate_api::HostTensorView {
            data: &x.data,
            shape: &x.shape,
        };
        let y_view = runmat_accelerate_api::HostTensorView {
            data: &y.data,
            shape: &y.shape,
        };
        let x_handle = trait_provider.upload(&x_view).expect("upload x");
        let y_handle = trait_provider.upload(&y_view).expect("upload y");

        let gpu_eval = evaluate(
            Value::GpuTensor(x_handle.clone()),
            Value::GpuTensor(y_handle.clone()),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[],
        )
        .expect("gpu polyfit");

        let _ = trait_provider.free(&x_handle);
        let _ = trait_provider.free(&y_handle);

        let cpu_coeff = match cpu_eval.coefficients() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor coefficients, got {other:?}"),
        };
        let gpu_coeff = match gpu_eval.coefficients() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor coefficients, got {other:?}"),
        };
        assert_eq!(cpu_coeff.shape, gpu_coeff.shape);
        for (a, b) in cpu_coeff.data.iter().zip(gpu_coeff.data.iter()) {
            assert!((a - b).abs() < 1e-9, "coeff mismatch {a} vs {b}");
        }

        let cpu_mu = match cpu_eval.mu() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor mu, got {other:?}"),
        };
        let gpu_mu = match gpu_eval.mu() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor mu, got {other:?}"),
        };
        assert_eq!(cpu_mu.shape, gpu_mu.shape);
        for (a, b) in cpu_mu.data.iter().zip(gpu_mu.data.iter()) {
            assert!((a - b).abs() < 1e-9, "mu mismatch {a} vs {b}");
        }

        let cpu_stats = match cpu_eval.stats() {
            Value::Struct(s) => s,
            other => panic!("expected struct stats, got {other:?}"),
        };
        let gpu_stats = match gpu_eval.stats() {
            Value::Struct(s) => s,
            other => panic!("expected struct stats, got {other:?}"),
        };
        let cpu_r = match cpu_stats.fields.get("R").expect("R present") {
            Value::Tensor(t) => t.clone(),
            other => panic!("expected tensor R, got {other:?}"),
        };
        let gpu_r = match gpu_stats.fields.get("R").expect("R present") {
            Value::Tensor(t) => t.clone(),
            other => panic!("expected tensor R, got {other:?}"),
        };
        assert_eq!(cpu_r.shape, gpu_r.shape);
        for (a, b) in cpu_r.data.iter().zip(gpu_r.data.iter()) {
            assert!((a - b).abs() < 1e-9, "R mismatch {a} vs {b}");
        }
        let cpu_df = match cpu_stats.fields.get("df").expect("df present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric df, got {other:?}"),
        };
        let gpu_df = match gpu_stats.fields.get("df").expect("df present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric df, got {other:?}"),
        };
        assert!((cpu_df - gpu_df).abs() < 1e-9);
        let cpu_normr = match cpu_stats.fields.get("normr").expect("normr present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric normr, got {other:?}"),
        };
        let gpu_normr = match gpu_stats.fields.get("normr").expect("normr present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric normr, got {other:?}"),
        };
        assert!((cpu_normr - gpu_normr).abs() < 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_mismatched_lengths() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            &[],
        )
        .expect_err("polyfit should reject mismatched vector lengths");
        assert_error_contains(err, "same length");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_vector_inputs() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            &[],
        )
        .expect_err("polyfit should reject non-vector X");
        assert_error_contains(err, "vector");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_weight_length_mismatch() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[Value::Tensor(weights)],
        )
        .expect_err("polyfit should reject mismatched weights");
        assert_error_contains(err, "weight vector must match");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_negative_weights() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, -1.0, 1.0], vec![3, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[Value::Tensor(weights)],
        )
        .expect_err("polyfit should reject negative weights");
        assert_error_contains(err, "non-negative");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fits_complex_data() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let complex_values =
            ComplexTensor::new(vec![(0.0, 1.0), (1.0, 0.5), (4.0, -0.25)], vec![3, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::ComplexTensor(complex_values),
            Value::Int(runmat_builtins::IntValue::I32(2)),
            &[],
        )
        .expect("polyfit complex");
        match eval.coefficients() {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
            }
            other => panic!("expected complex tensor coefficients, got {other:?}"),
        }
    }
}
