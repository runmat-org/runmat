//! MATLAB-compatible `polyfit` builtin with GPU-aware semantics for RunMat.

use log::warn;
use num_complex::Complex64;
use runmat_accelerate_api::{ProviderPolyfitOutputs, ProviderPolyfitResult};
#[cfg(test)]
use runmat_builtins::POLYFIT_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, PolyfitStatField, POLYFIT_ERROR_INTERNAL,
    POLYFIT_ERROR_INVALID_ARGUMENT, POLYFIT_ERROR_INVALID_INPUT, POLYFIT_ERROR_PROVIDER_FAILED,
    POLYFIT_INTEGER_WEIGHTS_EXTENSION, POLYFIT_INTEGER_X_EXTENSION, POLYFIT_INTEGER_Y_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::normalization;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::dispatcher;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

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

#[path = "polyfit/kernel_error.rs"]
pub mod kernel_error;
use kernel_error::{PolyfitKernelError, PolyfitKernelResult};
#[path = "polyfit/geometry.rs"]
mod geometry;
use geometry::SolverGeometry;
#[path = "polyfit/column_scale.rs"]
mod column_scale;
#[path = "polyfit/response_scale.rs"]
mod response_scale;
#[path = "polyfit/solver.rs"]
mod solver;
#[path = "polyfit/provider_output.rs"]
mod provider_output;
use provider_output::{failure as provider_output_failure, ProviderOutputFault};
#[cfg(test)]
use solver::{compute_scale, householder_qr, scaled_coordinates, solve_upper, stable_l2};
use solver::{solve_polyfit, validate_weights};

fn polyfit_error_with(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn polyfit_error(message: impl Into<String>) -> RuntimeError {
    polyfit_error_with(message, &POLYFIT_ERROR_INVALID_INPUT)
}

fn polyfit_argument_error(message: impl Into<String>) -> RuntimeError {
    polyfit_error_with(message, &POLYFIT_ERROR_INVALID_ARGUMENT)
}

fn polyfit_internal_error(message: impl Into<String>) -> RuntimeError {
    polyfit_error_with(message, &POLYFIT_ERROR_INTERNAL)
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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::poly::polyfit"
)]
async fn polyfit_builtin(
    x: Value,
    y: Value,
    degree: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let requested_count = crate::output_count::current_output_count();
    let outputs = requested_outputs(requested_count);
    let eval = evaluate_requested(x, y, degree, rest.view(), outputs).await?;
    if let Some(out_count) = requested_count {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        let mut outputs = vec![eval.coefficients()];
        if out_count >= 2 {
            outputs.push(eval.stats().ok_or_else(|| {
                polyfit_internal_error("polyfit: requested statistics were not produced")
            })?);
        }
        if out_count >= 3 {
            outputs.push(eval.mu());
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(eval.coefficients())
}

fn requested_outputs(requested_count: Option<usize>) -> ProviderPolyfitOutputs {
    match requested_count {
        Some(count) if count >= 3 => ProviderPolyfitOutputs::StatisticsAndNormalization,
        Some(count) if count >= 2 => ProviderPolyfitOutputs::Statistics,
        _ => ProviderPolyfitOutputs::Coefficients,
    }
}

/// Evaluate `polyfit` with statistics available to direct callers. The
/// registered binding uses the captured requested-output count instead.
pub async fn evaluate(
    x: Value,
    y: Value,
    degree: Value,
    rest: &[Value],
) -> BuiltinResult<PolyfitEval> {
    let count = crate::output_count::current_output_count()
        .unwrap_or(2)
        .max(2);
    let outputs = requested_outputs(Some(count));
    evaluate_requested(
        x,
        y,
        degree,
        crate::call::FunctionArgumentView::from_slice(rest),
        outputs,
    )
    .await
}

async fn evaluate_requested(
    x: Value,
    y: Value,
    degree: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    outputs: ProviderPolyfitOutputs,
) -> BuiltinResult<PolyfitEval> {
    if rest.len() > 1 {
        return Err(polyfit_argument_error("polyfit: too many input arguments"));
    }
    let deg = parse_degree(&degree)?;
    gpu_helpers::require_resident_compute_values(
        &[&x, &y, rest.first().unwrap_or(&x)],
        false,
        || polyfit_internal_error("polyfit: no exact owner for resident input"),
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&x, "polyfit")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&y, "polyfit")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&degree, "polyfit")?;
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, "polyfit")?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &POLYFIT_INTEGER_X_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &POLYFIT_INTEGER_Y_EXTENSION,
        BUILTIN_NAME,
        "Y",
    )
    .await?;
    if let Some(weights) = rest.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            weights,
            &POLYFIT_INTEGER_WEIGHTS_EXTENSION,
            BUILTIN_NAME,
            "weights",
        )
        .await?;
    }
    if let Some(eval) = try_gpu_polyfit(&x, &y, deg, rest, outputs).await? {
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
    if deg >= x_data.len() && x_data.len() > 1 {
        warn!(
            "polyfit: polynomial degree {} is ill-conditioned for {} data points; results may be inaccurate",
            deg,
            x_data.len()
        );
    }

    let weights = parse_weights(rest, x_data.len(), deg, outputs).await?;
    let mut solution = solve_polyfit(&x_data, &y_data, deg, weights.as_deref(), outputs)?;
    if is_complex_input {
        solution.is_complex = true;
    }

    PolyfitEval::from_solution(solution)
}

async fn try_gpu_polyfit(
    x: &Value,
    y: &Value,
    degree: usize,
    rest: crate::call::FunctionArgumentView<'_>,
    outputs: ProviderPolyfitOutputs,
) -> BuiltinResult<Option<PolyfitEval>> {
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
    let inputs = [Some(x_handle), Some(y_handle), weight_handle]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut count = None;
    for input in &inputs {
        ensure_vector_shape("polyfit", "input", input.shape())?;
        let len = if input.shape().contains(&0) {
            0
        } else {
            input
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| polyfit_error("polyfit: input size overflow"))?
        };
        if len == 0 || count.is_some_and(|count| count != len) {
            return Err(polyfit_error(
                "polyfit: input vectors must have matching nonempty lengths",
            ));
        }
        count = Some(len);
    }
    SolverGeometry::for_outputs(
        count.expect("x and y lengths were validated"),
        degree,
        outputs,
    )?;
    gpu_helpers::require_resident_compute_owners(&inputs, false, || {
        polyfit_internal_error("polyfit: no exact owner for resident input")
    })?;
    let provider = gpu_helpers::require_resident_compute_owner(x_handle, || {
        polyfit_internal_error("polyfit: no exact owner for resident input")
    })?;
    for input in &inputs {
        let owner = gpu_helpers::require_resident_compute_owner(input, || {
            polyfit_internal_error("polyfit: no exact owner for resident input")
        })?;
        if !runmat_accelerate_api::same_provider_instance(owner, provider)
            || input.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(input)
            || runmat_accelerate_api::handle_precision(input) != Some(provider.precision())
        {
            return Ok(None);
        }
    }
    let _residency = inputs
        .iter()
        .map(|input| gpu_helpers::ResidencyRestoreGuard::new(input))
        .collect::<Vec<_>>();

    let result = match provider
        .polyfit(x_handle, y_handle, degree, weight_handle, outputs)
        .await
    {
        Ok(res) => res,
        Err(error)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&error) =>
        {
            return Ok(None)
        }
        Err(error) => {
            return Err(kernel_error::with_provider_source(
                polyfit_internal_error(format!("polyfit: provider execution failed: {error}")),
                runmat_accelerate_api::ProviderExecutionError::from(error),
            ))
        }
    };
    let solution = PolyfitSolution::from_provider(
        result,
        outputs,
        degree,
        count.expect("resident input count validated"),
    )?;
    PolyfitEval::from_solution(solution).map(Some)
}

#[derive(Clone, Debug)]
struct PolyfitSolution {
    coeffs: Vec<Complex64>,
    r_matrix: Option<Vec<f64>>,
    mu_mean: f64,
    mu_scale: f64,
    normr: Option<f64>,
    df: f64,
    cols: usize,
    is_complex: bool,
}

impl PolyfitSolution {
    fn from_provider(
        result: ProviderPolyfitResult,
        outputs: ProviderPolyfitOutputs,
        degree: usize,
        rows: usize,
    ) -> BuiltinResult<Self> {
        let cols = result.coefficients.len();
        let expected_cols = degree.checked_add(1).ok_or_else(|| {
            provider_output_failure(ProviderOutputFault::CoefficientCount {
                expected: usize::MAX,
                observed: cols,
            })
        })?;
        if cols != expected_cols {
            return Err(provider_output_failure(
                ProviderOutputFault::CoefficientCount {
                    expected: expected_cols,
                    observed: cols,
                },
            ));
        }
        if let Some(index) = result
            .coefficients
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(provider_output_failure(
                ProviderOutputFault::NonFiniteCoefficient { index },
            ));
        }
        if result.r_matrix.is_some() != outputs.wants_statistics() {
            return Err(provider_output_failure(
                ProviderOutputFault::StatisticsPresence,
            ));
        }
        if result.normr.is_some() != outputs.wants_statistics() {
            return Err(provider_output_failure(
                ProviderOutputFault::ResidualPresence,
            ));
        }
        if let Some(r) = &result.r_matrix {
            let expected = cols.checked_mul(cols).ok_or_else(|| {
                provider_output_failure(ProviderOutputFault::RLength {
                    expected: usize::MAX,
                    observed: r.len(),
                })
            })?;
            if r.len() != expected {
                return Err(provider_output_failure(ProviderOutputFault::RLength {
                    expected,
                    observed: r.len(),
                }));
            }
            if let Some(index) = r.iter().position(|value| !value.is_finite()) {
                return Err(provider_output_failure(ProviderOutputFault::NonFiniteR {
                    index,
                }));
            }
        }
        if result
            .normr
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(provider_output_failure(
                ProviderOutputFault::NonFiniteResidual,
            ));
        }
        let expected_df = rows.saturating_sub(cols) as f64;
        if result.df != expected_df {
            return Err(provider_output_failure(
                ProviderOutputFault::DegreesOfFreedom {
                    expected: expected_df,
                    observed: result.df,
                },
            ));
        }
        let [mu_mean, mu_scale] = result.mu;
        let valid_mu = match outputs.basis() {
            runmat_accelerate_api::ProviderPolyfitBasis::Original => {
                mu_mean == 0.0 && mu_scale == 1.0
            }
            runmat_accelerate_api::ProviderPolyfitBasis::CenteredScaled => {
                mu_mean.is_finite() && mu_scale.is_finite() && mu_scale > 0.0
            }
        };
        if !valid_mu {
            return Err(provider_output_failure(
                ProviderOutputFault::InvalidNormalization,
            ));
        }
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
    stats: Option<Value>,
    mu: Value,
    is_complex: bool,
}

impl PolyfitEval {
    fn from_solution(solution: PolyfitSolution) -> BuiltinResult<Self> {
        let coefficients = coefficients_to_value(&solution.coeffs)?;
        let stats = match (&solution.r_matrix, solution.normr) {
            (Some(r), Some(normr)) => Some(build_stats(r, solution.cols, normr, solution.df)?),
            (None, None) => None,
            _ => {
                return Err(polyfit_internal_error(
                    "polyfit: QR statistics and residual norm must be returned together",
                ))
            }
        };
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

    /// Structure `S` containing fields `R`, `df`, and `normr`, if requested.
    pub fn stats(&self) -> Option<Value> {
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
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            polyfit_argument_error("polyfit: degree must be a non-negative integer")
        });
    }
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(polyfit_argument_error("polyfit: degree must be finite"));
            }
            let rounded = n.round();
            if rounded != *n {
                return Err(polyfit_argument_error("polyfit: degree must be an integer"));
            }
            if rounded < 0.0 {
                return Err(polyfit_argument_error(
                    "polyfit: degree must be a non-negative integer",
                ));
            }
            if !fits_platform_usize(rounded) {
                return Err(polyfit_argument_error(
                    "polyfit: degree exceeds platform limits",
                ));
            }
            Ok(rounded as usize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            parse_degree(&Value::Num(tensor::tensor_value_f64(t, 0)))
        }
        Value::LogicalArray(l) if l.len() == 1 => {
            parse_degree(&Value::Num(if l.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        other => Err(polyfit_argument_error(format!(
            "polyfit: degree must be a scalar numeric value, got {other:?}"
        ))),
    }
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

#[async_recursion::async_recursion(?Send)]
async fn real_vector(context: &str, label: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(context, label, tensor.shape())?;
            Ok(tensor::tensor_values_f64(&tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(polyfit_error)?;
            ensure_vector_shape(context, label, tensor.shape())?;
            Ok(tensor::tensor_values_f64(&tensor))
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
        Value::Tensor(tensor) => {
            ensure_vector_shape(context, label, tensor.shape())?;
            let all_real = true;
            let data = tensor::tensor_values_f64(&tensor)
                .into_iter()
                .map(|x| Complex64::new(x, 0.0))
                .collect();
            Ok((data, all_real))
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(context, label, tensor.shape())?;
            let values = tensor.materialize_f64();
            let is_complex = values.iter().any(|&(_, im)| im != 0.0);
            let data = values
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>();
            Ok((data, is_complex))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(polyfit_error)?;
            ensure_vector_shape(context, label, tensor.shape())?;
            Ok((
                tensor::tensor_values_f64(&tensor)
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect(),
                false,
            ))
        }
        Value::Num(n) => Ok((vec![Complex64::new(n, 0.0)], false)),
        Value::Int(i) => Ok((vec![Complex64::new(i.to_f64(), 0.0)], false)),
        Value::Bool(b) => Ok((vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)], false)),
        Value::Complex(re, im) => Ok((vec![Complex64::new(re, im)], im != 0.0)),
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

async fn parse_weights(
    rest: crate::call::FunctionArgumentView<'_>,
    len: usize,
    degree: usize,
    outputs: ProviderPolyfitOutputs,
) -> BuiltinResult<Option<Vec<f64>>> {
    match rest.len() {
        0 => Ok(None),
        1 => {
            let gathered = dispatcher::gather_if_needed_async(&rest[0]).await?;
            let data = real_vector("polyfit", "weights", gathered).await?;
            if data.len() != len {
                return Err(polyfit_argument_error(
                    "polyfit: weight vector must match the size of X",
                ));
            }
            SolverGeometry::for_outputs(len, degree, outputs)?;
            validate_weights(&data)?;
            Ok(Some(data))
        }
        _ => Err(polyfit_argument_error("polyfit: too many input arguments")),
    }
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

fn coefficients_to_value(coeffs: &[Complex64]) -> BuiltinResult<Value> {
    let all_real = coeffs.iter().all(|c| c.im == 0.0 && c.re.is_finite());
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
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.fields.insert(
        PolyfitStatField::R.public_name().to_string(),
        Value::Tensor(tensor),
    );
    st.fields.insert(
        PolyfitStatField::DegreesOfFreedom.public_name().to_string(),
        Value::Num(df),
    );
    st.fields.insert(
        PolyfitStatField::ResidualNorm.public_name().to_string(),
        Value::Num(normr),
    );
    Ok(Value::Struct(st.finish()))
}

fn build_mu(mean: f64, scale: f64) -> BuiltinResult<Value> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(polyfit_error("polyfit: mu(2) must be non-zero and finite"));
    }
    let tensor = Tensor::new(vec![mean, scale], vec![1, 2])
        .map_err(|e| polyfit_error(format!("polyfit: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[derive(Debug, Clone)]
pub struct PolyfitHostRealResult {
    pub coefficients: Vec<f64>,
    pub r_matrix: Option<Vec<f64>>,
    pub mu: [f64; 2],
    pub normr: Option<f64>,
    pub df: f64,
}

pub fn polyfit_host_real_for_provider(
    x: &[f64],
    y: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
    outputs: ProviderPolyfitOutputs,
) -> PolyfitKernelResult<PolyfitHostRealResult> {
    if x.len() != y.len() {
        return Err(PolyfitKernelError::input(
            "polyfit: X and Y vectors must be the same length",
        ));
    }
    if let Some(w) = weights {
        if w.len() != x.len() {
            return Err(PolyfitKernelError::input(
                "polyfit: weight vector must match the size of X",
            ));
        }
    }
    // Check the complex RHS conversion as well as the downstream matrix
    // allocations before allocating any provider-fallback workspace.
    if !x.is_empty() {
        SolverGeometry::for_outputs(x.len(), degree, outputs)?;
    }
    if let Some(w) = weights {
        validate_weights(w)?;
    }
    let complex_y: Vec<Complex64> = y.iter().copied().map(|v| Complex64::new(v, 0.0)).collect();
    let solution = solve_polyfit(x, &complex_y, degree, weights, outputs)?;
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
        return Err(PolyfitKernelError::input(
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
#[path = "polyfit/tests.rs"]
pub(crate) mod tests;
