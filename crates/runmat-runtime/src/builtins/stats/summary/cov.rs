//! MATLAB-compatible `cov` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    AccelProvider, CovNormalization, CovRows, CovarianceOptions, GpuTensorHandle, GpuTensorStorage,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, CovRowsOption, COV_ERROR_COMPLEX_UNSUPPORTED, COV_ERROR_INTERNAL,
    COV_ERROR_INVALID_ARGUMENT, COV_ERROR_NORMALIZATION_DUPLICATE, COV_ERROR_NORMALIZATION_INVALID,
    COV_ERROR_ROWS_MISMATCH, COV_ERROR_ROWS_OPTION_UNKNOWN, COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS,
    COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH, COV_INTEGER_DATA_EXTENSION,
    COV_LOGICAL_DATA_EXTENSION, COV_TYPED_NORMALIZATION_EXTENSION, COV_VECTOR_WEIGHTS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::stats::summary::covariance_precision::{
    AlignedCovarianceOutput, CovarianceAlignmentContract, CovariancePrecision,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cov";

fn cov_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cov_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cov_error_with(error, error.message)
}

fn cov_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    cov_error_with(error, format!("{}: {detail}", error.message))
}

fn cov_provider_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    let mut source: &(dyn std::error::Error + 'static) = &error;
    let mut weight_domain = false;
    loop {
        if source.is::<runmat_accelerate_api::CovarianceWeightDomainError>() {
            weight_domain = true;
            break;
        }
        let Some(next) = source.source() else { break };
        source = next;
    }
    let fallback = if weight_domain {
        cov_error_with_detail(&COV_ERROR_INVALID_ARGUMENT, &error)
    } else {
        cov_internal_error(error.to_string())
    };
    crate::runtime_error::with_provider_execution_source(fallback, error)
}

fn cov_internal_error(message: impl Into<String>) -> RuntimeError {
    cov_error_with(&COV_ERROR_INTERNAL, message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cov",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("covariance")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "GPU execution is available when rows='all' and no weight vector is supplied; other cases fall back to the CPU path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cov",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "The covariance builtin is treated as a fusion boundary and executes via dedicated kernels or the host reference.",
};

#[runtime_builtin(
    name = "cov",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::cov"
)]
async fn cov_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let _argument_sources = rest.fork();
    let args = CovArgs::parse(value, rest)?;
    ensure_cov_extensions(&args)?;
    let precision = CovariancePrecision::observations(&args.first, args.second.as_ref());
    // Preserve weight admission precedence while rejecting invalid metadata
    // before any upload, download, or covariance hook.
    if let (Some(input), Some(weights)) = (
        covariance_input_shape(&args.first),
        args.weight_vector.as_ref().and_then(covariance_input_shape),
    ) {
        validate_weight_shape(
            weights,
            covariance_observation_count(input, args.second.is_some())?,
        )?;
    }
    if let (Some(left), Some(right)) = (
        covariance_input_shape(&args.first),
        args.second.as_ref().and_then(covariance_input_shape),
    ) {
        validate_covariance_pair_shapes(left, right)?;
    }
    // Resolve every borrowed resident operand before any hook or host gather.
    // The legacy lookup may return an unrelated ambient provider.
    for handle in std::iter::once(&args.first)
        .chain(args.second.as_ref())
        .chain(args.weight_vector.as_ref())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
    {
        gpu_helpers::require_resident_compute_owner(handle, || {
            cov_internal_error("cov: resident operand has no exact owning provider")
        })?;
        handle.allocation_lease().validate_live().map_err(|error| {
            cov_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                error,
            ))
        })?;
    }
    if let Some(result) = cov_try_gpu(&args).await? {
        return Ok(result);
    }
    let resident_inputs: Vec<_> = std::iter::once(&args.first)
        .chain(args.second.as_ref())
        .chain(args.weight_vector.as_ref())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let output_source =
        gpu_helpers::select_resident_output_source(resident_inputs.iter().cloned(), NAME)?;
    // cov_host consumes the argument values. Keep all original leases and
    // intent alive through every download, suspension and output restoration.
    let protected: Vec<_> = resident_inputs.iter().collect();
    let _residency: Vec<_> = protected
        .iter()
        .map(|handle| gpu_helpers::ResidencyRestoreGuard::new(handle))
        .collect();
    let result = precision.cast_host(cov_host(args).await?, NAME)?;
    match output_source {
        Some(source) => gpu_helpers::restore_class_preserving_value_protecting(
            &source, result, NAME, &protected,
        ),
        None => Ok(result),
    }
}

/// Public entry point for providers that need the reference implementation.
pub fn cov_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    rows: CovRows,
    weight: CovWeightSpec,
) -> BuiltinResult<Tensor> {
    let matrix = combine_tensors(left, right)?;
    if let CovWeightSpec::Vector(ref vec) = weight {
        if matrix.rows != vec.len() {
            return Err(cov_error_with_detail(
                &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
                format!("expected {} elements", matrix.rows),
            ));
        }
    }
    match rows {
        CovRows::All => covariance_dense(&matrix, &weight),
        CovRows::OmitRows => {
            let (filtered, filtered_weight) = filter_complete_rows(&matrix, weight);
            covariance_dense(&filtered, &filtered_weight)
        }
        CovRows::PartialRows => covariance_pairwise(&matrix, &weight),
    }
}

mod admission;
use admission::{ensure_cov_extensions, value_rows_cols, CovArgs};

#[derive(Debug, Clone)]
pub enum CovWeightSpec {
    Scalar(CovNormalization),
    Vector(Vec<f64>),
}

mod provider;
use provider::{cov_try_gpu, covariance_observation_count, validate_weight_shape};

mod host;
use host::cov_host;

mod domain;
use domain::{
    combine_tensors, covariance_column_count, covariance_dense, covariance_element_count,
    covariance_input_shape, covariance_pairwise, filter_complete_rows,
    validate_covariance_pair_shapes,
};

#[cfg(test)]
pub(crate) mod tests;
