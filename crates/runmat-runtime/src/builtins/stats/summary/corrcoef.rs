//! MATLAB-compatible `corrcoef` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::{
    CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, GpuTensorStorage,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, CorrcoefOptionKey, CorrcoefRowsOption, CORRCOEF_ERROR_ALPHA_INVALID,
    CORRCOEF_ERROR_COMPLEX_OUTPUTS, CORRCOEF_ERROR_INTERNAL, CORRCOEF_ERROR_INVALID_ARGUMENT,
    CORRCOEF_ERROR_OPTION_DUPLICATE, CORRCOEF_ERROR_OPTION_UNKNOWN,
    CORRCOEF_ERROR_ROWS_OPTION_MALFORMED, CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN,
    CORRCOEF_ERROR_SIZE_MISMATCH, CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS,
    CORRCOEF_INTEGER_DATA_EXTENSION, CORRCOEF_LOGICAL_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor::{self, value_to_string};
use crate::builtins::stats::summary::covariance_precision::{
    AlignedCovarianceOutput, CovarianceAlignmentContract, CovariancePrecision,
};
use crate::builtins::stats::summary::distribution_math::{
    standard_normal_inv, student_t_cdf_upper,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "corrcoef";

fn corrcoef_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn corrcoef_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    corrcoef_error_with(error, error.message)
}

fn corrcoef_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    corrcoef_error_with(error, format!("{}: {detail}", error.message))
}

fn corrcoef_provider_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        corrcoef_internal_error(error.to_string()),
        error,
    )
}

fn corrcoef_internal_error(message: impl Into<String>) -> RuntimeError {
    corrcoef_error_with(&CORRCOEF_ERROR_INTERNAL, message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "corrcoef",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("corrcoef")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side corrcoef kernels when rows='all'; other cases fall back to host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "corrcoef",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner treats corrcoef as a non-fusible boundary; GPU execution is provided via a custom provider hook.",
};

#[runtime_builtin(
    name = "corrcoef",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::corrcoef"
)]
async fn corrcoef_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let _argument_sources = rest.fork();
    let args = CorrcoefArgs::parse(value, rest)?;
    ensure_corrcoef_extensions(&args)?;
    if crate::output_count::current_output_count() != Some(0) {
        let values = [&args.first, args.second.as_ref().unwrap_or(&args.first)];
        gpu_helpers::require_resident_compute_values(&values, false, || {
            corrcoef_internal_error("corrcoef: resident operand has no exact owning provider")
        })?;
    }
    admit_corrcoef_inputs(&args)?;
    let precision = CovariancePrecision::observations(&args.first, args.second.as_ref());
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    if requested_outputs.is_none() || requested_outputs == Some(1) {
        if let Some(result) = corrcoef_try_gpu(&args).await? {
            return Ok(match requested_outputs {
                Some(1) => Value::output_list(vec![result]),
                _ => result,
            });
        }
    }
    precision.cast_host(corrcoef_host(args, requested_outputs).await?, NAME)
}

/// Exposed for acceleration providers that need the host reference implementation.
pub fn corrcoef_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    _normalization: CorrcoefNormalization,
    rows: CorrcoefRows,
) -> BuiltinResult<Tensor> {
    match evaluate_inputs(
        NumericInput::Real(left),
        right.map(NumericInput::Real),
        rows,
    )? {
        CorrcoefEvaluation::Real(evaluation) => Ok(evaluation.r),
        CorrcoefEvaluation::Complex(_) => {
            unreachable!("real tensor inputs cannot produce a complex result")
        }
    }
}

mod domain;
mod admission;
mod host;
mod provider;
use admission::{admit_corrcoef_inputs, ensure_corrcoef_extensions, CorrcoefArgs};
use domain::{evaluate_inputs, same_size, CorrcoefEvaluation, NumericInput};
use host::corrcoef_host;
use provider::corrcoef_try_gpu;

#[cfg(test)]
pub(crate) mod tests;
