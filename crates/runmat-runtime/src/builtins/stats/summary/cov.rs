//! MATLAB-compatible `cov` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    AccelProvider, CovNormalization, CovRows, CovarianceOptions, GpuTensorHandle, GpuTensorStorage,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, CovRowsOption, COV_DESCRIPTOR, COV_ERROR_COMPLEX_UNSUPPORTED,
    COV_ERROR_INTERNAL, COV_ERROR_INVALID_ARGUMENT, COV_ERROR_NORMALIZATION_DUPLICATE,
    COV_ERROR_NORMALIZATION_INVALID, COV_ERROR_ROWS_MISMATCH, COV_ERROR_ROWS_OPTION_UNKNOWN,
    COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS, COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
    COV_INTEGER_DATA_EXTENSION, COV_LOGICAL_DATA_EXTENSION, COV_TYPED_NORMALIZATION_EXTENSION,
    COV_VECTOR_WEIGHTS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
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
async fn cov_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = CovArgs::parse(value, rest)?;
    ensure_cov_extensions(&args)?;
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
        if gpu_helpers::exact_provider_for_handle(handle).is_none() {
            let mut error =
                cov_internal_error("cov: resident operand has no exact owning provider");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
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
    let result = cov_host(args).await?;
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

#[derive(Debug)]
struct CovArgs {
    first: Value,
    second: Option<Value>,
    normalization: CovNormalization,
    rows: CovRows,
    weight_vector: Option<Value>,
    typed_normalization: bool,
}

impl CovArgs {
    fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut second_candidate: Option<Value> = None;
        let mut weight_candidate: Option<Value> = None;
        let mut normalization = CovNormalization::Unbiased;
        let mut normalization_explicit = false;
        let mut typed_normalization = false;
        let mut rows = CovRows::All;

        let iter = rest.into_iter();
        for arg in iter {
            match arg {
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                    let key = tensor::value_to_string(&arg)
                        .ok_or_else(|| cov_error(&COV_ERROR_INVALID_ARGUMENT))?;
                    let lowered = key.trim().to_ascii_lowercase();
                    rows = parse_rows_option(&lowered)?;
                }
                Value::Tensor(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => {
                    if second_candidate.is_none() {
                        second_candidate = Some(arg);
                    } else if weight_candidate.is_none() {
                        weight_candidate = Some(arg);
                    } else {
                        return Err(cov_error(&COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS));
                    }
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                    if normalization_explicit || weight_candidate.is_some() {
                        return Err(cov_error(&COV_ERROR_NORMALIZATION_DUPLICATE));
                    }
                    typed_normalization = matches!(arg, Value::Int(_) | Value::Bool(_));
                    normalization = parse_normalization(arg)?;
                    normalization_explicit = true;
                }
                Value::ComplexTensor(_) => {
                    return Err(cov_error(&COV_ERROR_COMPLEX_UNSUPPORTED));
                }
                other => {
                    return Err(cov_error_with_detail(
                        &COV_ERROR_INVALID_ARGUMENT,
                        format!("{other:?}"),
                    ))
                }
            }
        }

        if let Some(weight_array) = weight_candidate {
            // Explicit weight vector always takes precedence over dataset detection.
            return Ok(Self {
                first,
                second: second_candidate,
                normalization,
                rows,
                weight_vector: Some(weight_array),
                typed_normalization,
            });
        }

        let mut second = second_candidate;
        let mut weight_vector: Option<Value> = None;

        if let Some(candidate) = second.take() {
            if should_treat_as_weight(&first, &candidate, normalization_explicit, rows)? {
                weight_vector = Some(candidate);
            } else {
                second = Some(candidate);
            }
        }

        Ok(Self {
            first,
            second,
            normalization,
            rows,
            weight_vector,
            typed_normalization,
        })
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        )
}

fn ensure_cov_extensions(args: &CovArgs) -> BuiltinResult<()> {
    if is_typed_integer_value(&args.first)
        || args.second.as_ref().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_INTEGER_DATA_EXTENSION, NAME)?;
    }
    if is_logical_value(&args.first) || args.second.as_ref().is_some_and(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_LOGICAL_DATA_EXTENSION, NAME)?;
    }
    if args.typed_normalization {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_TYPED_NORMALIZATION_EXTENSION,
            NAME,
        )?;
    }
    if args.weight_vector.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_VECTOR_WEIGHTS_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub enum CovWeightSpec {
    Scalar(CovNormalization),
    Vector(Vec<f64>),
}

async fn cov_try_gpu(args: &CovArgs) -> BuiltinResult<Option<Value>> {
    if args.rows != CovRows::All {
        return Ok(None);
    }

    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let provider = match gpu_helpers::exact_provider_for_handle(first_handle) {
        Some(p) => p,
        None => return Ok(None),
    };

    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => {
            let Some(second_provider) = gpu_helpers::exact_provider_for_handle(handle) else {
                return Ok(None);
            };
            if !runmat_accelerate_api::same_provider_instance(provider, second_provider) {
                return Ok(None);
            }
            Some(handle)
        }
        Some(_) => return Ok(None),
        None => None,
    };

    let rows = covariance_observation_count(first_handle.shape(), maybe_second_handle.is_some())?;
    if let Some(second) = maybe_second_handle {
        validate_covariance_pair_shapes(first_handle.shape(), second.shape())?;
    }
    let protected: Vec<_> = std::iter::once(first_handle)
        .chain(maybe_second_handle)
        .collect();
    let _residency: Vec<_> = protected
        .iter()
        .map(|handle| crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(handle))
        .collect();
    let weight_handle =
        materialize_gpu_weight_vector(provider, args.weight_vector.as_ref(), rows, &protected)
            .await?;
    if args.weight_vector.is_some() && weight_handle.is_none() {
        return Ok(None);
    }
    let _weight_residency = match args.weight_vector.as_ref() {
        Some(Value::GpuTensor(handle)) => Some(gpu_helpers::ResidencyRestoreGuard::new(handle)),
        _ => None,
    };

    let options = CovarianceOptions {
        normalization: args.normalization,
        rows: args.rows,
        has_weight_vector: weight_handle.is_some(),
    };

    match provider
        .covariance(
            first_handle,
            maybe_second_handle,
            weight_handle.as_ref(),
            &options,
        )
        .await
    {
        Ok(mut result) => {
            let columns = if maybe_second_handle.is_some() {
                2
            } else {
                let (rows, columns) = value_rows_cols(&args.first)?;
                covariance_column_count(rows, columns)
            };
            let borrowed: Vec<_> = protected
                .iter()
                .copied()
                .chain(weight_handle.as_ref())
                .collect();
            // Both native floating precisions have existing covariance routes.
            // Exact Single/mixed precision reconciliation remains separate from
            // this ownership boundary; never relabel the producer's class.
            let valid = [
                runmat_accelerate_api::ProviderPrecision::F32,
                runmat_accelerate_api::ProviderPrecision::F64,
            ]
            .into_iter()
            .any(|precision| {
                gpu_helpers::binary_gpu_output_matches(
                    &result,
                    first_handle,
                    maybe_second_handle.unwrap_or(first_handle),
                    provider,
                    &gpu_helpers::BinaryGpuOutputContract {
                        shape: vec![columns, columns],
                        storage: runmat_accelerate_api::GpuTensorStorage::Real,
                        precision: Some(precision),
                        integer: None,
                        logical: false,
                        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
                    },
                )
            }) && borrowed
                .iter()
                .all(|input| !result.shares_allocation_with(input));
            if !valid {
                let mut error = cov_internal_error("cov: invalid provider covariance output");
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            if let Err(error) = result.allocation_lease().validate_live() {
                let error = cov_provider_error(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                );
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            gpu_helpers::propagate_output_provenance(&mut result, borrowed).map_err(|error| {
                cov_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                    error,
                ))
            })?;
            gpu_helpers::resident_gpu_value(result)
                .map(Some)
                .map_err(|error| {
                    cov_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                        error,
                    ))
                })
        }
        Err(error) => {
            if error.is_unsupported() {
                Ok(None)
            } else {
                Err(cov_provider_error(error))
            }
        }
    }
}

fn covariance_observation_count(shape: &[usize], paired_input: bool) -> BuiltinResult<usize> {
    if shape.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "inputs must be 2-D matrices or vectors",
        ));
    }
    let len = covariance_element_count(shape)?;
    Ok(match shape {
        [] => 1,
        _ if paired_input => len,
        [length] => *length,
        [rows, cols] if *rows == 1 || *cols == 1 => len,
        [rows, _] => *rows,
        _ => unreachable!("shape rank was checked above"),
    })
}

async fn materialize_gpu_weight_vector(
    provider: &dyn AccelProvider,
    value: Option<&Value>,
    expected_rows: usize,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if expected_rows == 0 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector cannot be empty",
        ));
    }

    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
            {
                return Ok(None);
            }
            validate_gpu_weight_shape(handle, expected_rows)?;
            let Some(weight_provider) = gpu_helpers::exact_provider_for_handle(handle) else {
                return Ok(None);
            };
            if !runmat_accelerate_api::same_provider_instance(provider, weight_provider) {
                return Ok(None);
            }
            Ok(Some(handle.clone()))
        }
        other => {
            let weights = value_to_weight_vector(other.clone(), expected_rows).await?;
            let shape = vec![expected_rows, 1];
            let weights = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => Tensor::from_f32(
                    weights.into_iter().map(|value| value as f32).collect(),
                    shape,
                ),
                runmat_accelerate_api::ProviderPrecision::F64 => Tensor::new(weights, shape),
            }
            .map_err(cov_internal_error)?;
            let handle = match crate::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
                provider, &weights, protected,
            ) {
                Ok(handle) => handle,
                Err(runmat_accelerate_api::NumericUploadError::Provider(error))
                    if gpu_helpers::provider_hook_is_unsupported(&error) =>
                {
                    return Ok(None)
                }
                Err(error) => {
                    return Err(crate::build_runtime_error(format!(
                        "cov: weight upload failed: {error}"
                    ))
                    .with_builtin(NAME)
                    .with_identifier(COV_ERROR_INTERNAL.identifier.expect("internal identifier"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build())
                }
            };
            Ok(Some(handle))
        }
    }
}

fn validate_gpu_weight_shape(handle: &GpuTensorHandle, expected_rows: usize) -> BuiltinResult<()> {
    validate_weight_shape(handle.shape(), expected_rows)
}

fn validate_weight_shape(shape: &[usize], expected_rows: usize) -> BuiltinResult<()> {
    if shape.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector must be one-dimensional",
        ));
    }
    let count = covariance_element_count(shape)?;
    if count == 0 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector cannot be empty",
        ));
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows != 1 && cols != 1 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector must be one-dimensional",
        ));
    }
    if count != expected_rows {
        return Err(cov_error_with_detail(
            &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
            format!("expected {expected_rows} elements"),
        ));
    }
    Ok(())
}

async fn cov_host(args: CovArgs) -> BuiltinResult<Value> {
    let CovArgs {
        first,
        second,
        normalization,
        rows,
        weight_vector,
        ..
    } = args;

    let left = value_to_tensor_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value).await?),
        None => None,
    };
    let expected_weight_rows = if right.is_some() || left.rows() == 1 || left.cols() == 1 {
        left.len()
    } else {
        left.rows()
    };

    let weight_spec = if let Some(weight_value) = weight_vector {
        let vector = value_to_weight_vector(weight_value, expected_weight_rows).await?;
        CovWeightSpec::Vector(vector)
    } else {
        CovWeightSpec::Scalar(normalization)
    };

    let tensor = cov_from_tensors(left, right, rows, weight_spec)?;
    Ok(Value::Tensor(tensor))
}

async fn value_to_tensor_gather(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error),
    }
}

async fn value_to_weight_vector(value: Value, expected_rows: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)?
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error)?,
    };
    validate_weight_shape(&tensor.shape, expected_rows)?;
    let values = tensor::tensor_into_values_f64(tensor);
    for (idx, weight) in values.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                format!("weights must be non-negative finite values (index {idx})"),
            ));
        }
    }
    Ok(values)
}

fn parse_rows_option(value: &str) -> BuiltinResult<CovRows> {
    match CovRowsOption::parse(value) {
        Some(CovRowsOption::All) => Ok(CovRows::All),
        Some(CovRowsOption::OmitRows) => Ok(CovRows::OmitRows),
        Some(CovRowsOption::PartialRows) => Ok(CovRows::PartialRows),
        None => Err(cov_error_with_detail(
            &COV_ERROR_ROWS_OPTION_UNKNOWN,
            format!("'{value}'"),
        )),
    }
}

fn parse_normalization(value: Value) -> BuiltinResult<CovNormalization> {
    match value {
        Value::Int(i) => match i.to_i128() {
            0 => Ok(CovNormalization::Unbiased),
            1 => Ok(CovNormalization::Biased),
            other => Err(cov_error_with_detail(
                &COV_ERROR_NORMALIZATION_INVALID,
                format!("expected 0 or 1, received {other}"),
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1.0e-12 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be an integer",
                ));
            }
            match rounded as i64 {
                0 => Ok(CovNormalization::Unbiased),
                1 => Ok(CovNormalization::Biased),
                other => Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    format!("expected 0 or 1, received {other}"),
                )),
            }
        }
        Value::Bool(flag) => Ok(if flag {
            CovNormalization::Biased
        } else {
            CovNormalization::Unbiased
        }),
        other => Err(cov_error_with_detail(
            &COV_ERROR_NORMALIZATION_INVALID,
            format!("value must be numeric, received {other:?}"),
        )),
    }
}

fn should_treat_as_weight(
    first: &Value,
    candidate: &Value,
    normalization_explicit: bool,
    rows_option: CovRows,
) -> BuiltinResult<bool> {
    let (rows_first, cols_first) = value_rows_cols(first)?;
    let (rows_candidate, cols_candidate) = value_rows_cols(candidate)?;

    let is_vector = rows_candidate == 1
        || cols_candidate == 1
        || rows_candidate * cols_candidate == rows_candidate
            && (rows_candidate == rows_first || cols_candidate == rows_first);

    if !is_vector {
        return Ok(false);
    }

    if rows_candidate != rows_first && cols_candidate != rows_first {
        // Length mismatch, treat as dataset so the later validation emits the proper error.
        return Ok(false);
    }

    if (rows_first == 1 || cols_first == 1)
        && !normalization_explicit
        && matches!(rows_option, CovRows::All)
    {
        // Ambiguous `cov(x, y)` case – prefer dataset semantics for compatibility.
        return Ok(false);
    }

    Ok(true)
}

fn value_rows_cols(value: &Value) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::Tensor(tensor) => Ok((tensor.rows(), tensor.cols())),
        Value::LogicalArray(array) => {
            if array.shape.len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if array.shape.is_empty() {
                1
            } else {
                array.shape[0]
            };
            let cols = if array.shape.len() >= 2 {
                array.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::GpuTensor(handle) => {
            if handle.shape().len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if handle.shape().is_empty() {
                1
            } else {
                handle.shape()[0]
            };
            let cols = if handle.shape().len() >= 2 {
                handle.shape()[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok((1, 1)),
        other => Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            format!("unsupported input type for shape inspection: {other:?}"),
        )),
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

// Shared host/resident result geometry: vectors and zero-column inputs are
// normalized to one variable; ordinary matrices retain their column count.
fn covariance_column_count(rows: usize, columns: usize) -> usize {
    if rows == 1 || columns <= 1 {
        1
    } else {
        columns
    }
}

fn covariance_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| cov_error_with_detail(&COV_ERROR_INVALID_ARGUMENT, "input shape overflow"))
}

fn covariance_input_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(&tensor.shape),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(&[1, 1]),
        _ => None,
    }
}

fn validate_covariance_pair_shapes(left: &[usize], right: &[usize]) -> BuiltinResult<()> {
    if left.len() > 2 || right.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "inputs must be 2-D matrices or vectors",
        ));
    }
    let vector = |shape: &[usize]| {
        shape.first().copied().unwrap_or(1) == 1 || shape.get(1).copied().unwrap_or(1) == 1
    };
    let compatible = if vector(left) && vector(right) {
        covariance_element_count(left)? == covariance_element_count(right)?
    } else {
        left == right
    };
    if compatible {
        Ok(())
    } else {
        Err(cov_error(&COV_ERROR_ROWS_MISMATCH))
    }
}

impl Matrix {
    fn from_single_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                "inputs must be 2-D matrices or vectors",
            ));
        }
        let original_rows = tensor.rows();
        let original_cols = tensor.cols();
        let is_vector = original_rows == 1 || original_cols == 1;
        let (rows, cols) = if is_vector {
            (
                tensor.len(),
                covariance_column_count(original_rows, original_cols),
            )
        } else if tensor.is_empty() && original_cols == 0 {
            (0, covariance_column_count(original_rows, original_cols))
        } else {
            (
                original_rows,
                covariance_column_count(original_rows, original_cols),
            )
        };
        Ok(Self {
            rows,
            cols,
            data: centered_tensor_values(&tensor, rows, cols),
        })
    }

    fn from_tensor_pair(left: Tensor, right: Tensor) -> BuiltinResult<Self> {
        validate_covariance_pair_shapes(&left.shape, &right.shape)?;
        let rows = left.len();
        let mut data = centered_tensor_values(&left, rows, 1);
        data.extend(centered_tensor_values(&right, rows, 1));
        Ok(Self {
            data,
            rows,
            cols: 2,
        })
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    fn column(&self, col: usize) -> &[f64] {
        let start = col * self.rows;
        let end = start + self.rows;
        &self.data[start..end]
    }
}

fn combine_tensors(left: Tensor, right: Option<Tensor>) -> BuiltinResult<Matrix> {
    if let Some(right) = right {
        Matrix::from_tensor_pair(left, right)
    } else {
        Matrix::from_single_tensor(left)
    }
}

fn centered_tensor_values(tensor: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor.materialize_f64();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let anchor = storage
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let value = storage
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(anchor);
            values.push((value - anchor) as f64);
        }
    }
    values
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn covariance_dense(matrix: &Matrix, weight: &CovWeightSpec) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    let rows = matrix.rows;

    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }

    let mut result = vec![f64::NAN; cols * cols];

    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let denom = match normalization {
                CovNormalization::Unbiased => ((rows as f64) - 1.0).max(1.0),
                CovNormalization::Biased => rows as f64,
            };
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut sum = 0.0;
                let mut valid = true;
                for &value in column {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    sum += value;
                }
                *mean_slot = if valid { sum / (rows as f64) } else { f64::NAN };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_unweighted_pair(matrix, i, j, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
        CovWeightSpec::Vector(weights) => {
            if weights.len() != rows {
                return Err(cov_error_with_detail(
                    &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
                    format!("expected {rows} elements"),
                ));
            }
            let sum_w: f64 = weights.iter().sum();
            if sum_w <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }
            let denom = sum_w - 1.0;
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut weighted_sum = 0.0;
                let mut valid = true;
                for (row, &value) in column.iter().enumerate() {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    weighted_sum += weights[row] * value;
                }
                *mean_slot = if valid {
                    weighted_sum / sum_w
                } else {
                    f64::NAN
                };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_weighted_pair(matrix, i, j, weights, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
    }

    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

fn filter_complete_rows(matrix: &Matrix, weight: CovWeightSpec) -> (Matrix, CovWeightSpec) {
    if matrix.rows == 0 {
        return (
            Matrix {
                data: Vec::new(),
                rows: 0,
                cols: matrix.cols,
            },
            weight,
        );
    }

    let mut valid_rows = Vec::new();
    for row in 0..matrix.rows {
        let mut is_valid = true;
        for col in 0..matrix.cols {
            if !matrix.get(row, col).is_finite() {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            valid_rows.push(row);
        }
    }

    if valid_rows.len() == matrix.rows {
        // No filtering required.
        return (matrix.clone(), weight);
    }

    let mut data = Vec::with_capacity(valid_rows.len() * matrix.cols);
    for col in 0..matrix.cols {
        for &row in &valid_rows {
            data.push(matrix.get(row, col));
        }
    }

    let filtered_matrix = Matrix {
        data,
        rows: valid_rows.len(),
        cols: matrix.cols,
    };

    let filtered_weight = match weight {
        CovWeightSpec::Scalar(norm) => CovWeightSpec::Scalar(norm),
        CovWeightSpec::Vector(vec) => {
            let mut filtered = Vec::with_capacity(valid_rows.len());
            for &row in &valid_rows {
                filtered.push(vec[row]);
            }
            CovWeightSpec::Vector(filtered)
        }
    };

    (filtered_matrix, filtered_weight)
}

fn covariance_pairwise(matrix: &Matrix, weight: &CovWeightSpec) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }
    let mut result = vec![f64::NAN; cols * cols];
    for i in 0..cols {
        let variance = covariance_pair(matrix, i, i, weight);
        set_entry(&mut result, cols, i, i, sanitize_covariance(true, variance));
        for j in (i + 1)..cols {
            let value = covariance_pair(matrix, i, j, weight);
            set_entry(&mut result, cols, i, j, sanitize_covariance(false, value));
        }
    }
    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

fn covariance_unweighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_weighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    weights: &[f64],
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
        if weight == 0.0 {
            continue;
        }
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += weight * (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_pair(matrix: &Matrix, lhs: usize, rhs: usize, weight: &CovWeightSpec) -> f64 {
    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for row in 0..matrix.rows {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                }
            }
            covariance_unweighted_slice(&xs, &ys, *normalization)
        }
        CovWeightSpec::Vector(weights) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut ws = Vec::new();
            for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                    ws.push(weight);
                }
            }
            covariance_weighted_slice(&xs, &ys, &ws)
        }
    }
}

fn covariance_unweighted_slice(xs: &[f64], ys: &[f64], normalization: CovNormalization) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    let denom = match normalization {
        CovNormalization::Unbiased => (n as f64) - 1.0,
        CovNormalization::Biased => n as f64,
    };
    if denom <= 0.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().take(n).sum();
    let sum_y: f64 = ys.iter().take(n).sum();
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn covariance_weighted_slice(xs: &[f64], ys: &[f64], weights: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() || weights.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len()).min(weights.len());
    if n == 0 {
        return f64::NAN;
    }
    let sum_w: f64 = weights.iter().take(n).sum();
    if sum_w <= 0.0 {
        return f64::NAN;
    }
    let denom = sum_w - 1.0;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for idx in 0..n {
        mean_x += weights[idx] * xs[idx];
        mean_y += weights[idx] * ys[idx];
    }
    mean_x /= sum_w;
    mean_y /= sum_w;
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += weights[idx] * (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn sanitize_covariance(is_diag: bool, value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if is_diag && value < 0.0 && value > -1.0e-12 {
        0.0
    } else {
        value
    }
}

fn set_entry(buffer: &mut [f64], dim: usize, row: usize, col: usize, value: f64) {
    let idx = row + col * dim;
    buffer[idx] = value;
    if row != col {
        let symmetrical = col + row * dim;
        buffer[symmetrical] = value;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
        let dim = (expected.len() as f64).sqrt() as usize;
        assert_eq!(actual.shape, vec![dim, dim], "unexpected tensor shape");
        for (idx, (&got, &want)) in actual
            .materialize_f64()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            if want.is_nan() {
                assert!(
                    got.is_nan(),
                    "expected NaN at linear index {idx}, found {got}"
                );
            } else {
                assert!(
                    (got - want).abs() <= tol,
                    "mismatch at linear index {idx}: got {got}, expected {want}"
                );
            }
        }
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
        Tensor::new_integer(storage, shape).unwrap()
    }

    #[test]
    fn cov_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = COV_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = cov(X)"));
        assert!(labels.contains(&"C = cov(X, normalization)"));
        assert!(labels.contains(&"C = cov(X, Y, w, opt)"));
    }

    #[cfg(feature = "wgpu")]
    fn cov_builtin_sync(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cov_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_matrix_basic() {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2, //
                0.60, 0.59, 0.58, 0.62, 0.63,
            ],
            vec![5, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(Value::Tensor(tensor), Vec::new())).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.0250, 0.0075, 0.00175, //
            0.0075, 0.0070, 0.00135, //
            0.00175, 0.00135, 0.00043,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_two_vectors() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![10.0, 11.0, 9.0, 12.0], vec![4, 1]).unwrap();
        let result = block_on(cov_builtin(Value::Tensor(x), vec![Value::Tensor(y)])).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            1.6666666666666667,
            0.6666666666666666, //
            0.6666666666666666,
            1.6666666666666667,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[test]
    fn cov_row_vector_scalar_empty_and_paired_matrix_shapes_match_contract() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(row_cov) =
            block_on(cov_builtin(Value::Tensor(row), Vec::new())).expect("row covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&row_cov, &[1.0], 1.0e-12);

        let scalar = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
        let Value::Tensor(scalar_cov) =
            block_on(cov_builtin(Value::Tensor(scalar), Vec::new())).expect("scalar covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&scalar_cov, &[0.0], 1.0e-12);

        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let Value::Tensor(empty_cov) =
            block_on(cov_builtin(Value::Tensor(empty), Vec::new())).expect("empty covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&empty_cov, &[f64::NAN], 0.0);

        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let Value::Tensor(pair_cov) =
            block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
                .expect("paired matrix covariance")
        else {
            panic!("expected tensor");
        };
        let expected = [5.0 / 3.0; 4];
        assert_tensor_close(&pair_cov, &expected, 1.0e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weighted_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.022380952380952376,
            0.004999999999999994, //
            0.004999999999999994,
            0.006666666666666678,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_accepts_typed_integer_matrix_and_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = poisoned_int_tensor(
            IntegerStorage::I16(vec![
                4, 4, 3, 4, 4, //
                2, 2, 2, 2, 2,
            ]),
            vec![5, 2],
            f64::NAN,
        );
        let weights = poisoned_int_tensor(
            IntegerStorage::U16(vec![1, 1, 1, 2, 2]),
            vec![5, 1],
            f64::NAN,
        );
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            1.0 / 7.0,
            0.0, //
            0.0,
            0.0,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_rejects_negative_typed_integer_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], 0.0);
        let weights = poisoned_int_tensor(IntegerStorage::I16(vec![1, -1]), vec![2, 1], 1.0);
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect_err("negative weights should fail");
        assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn cov_extensions_are_independently_gated_before_computation() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_error = block_on(cov_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap(),
            ),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            integer_error.identifier(),
            Some("RunMat:compatibility:CovIntegerDataExtension")
        );

        let logical_error = block_on(cov_builtin(Value::Bool(true), Vec::new())).unwrap_err();
        assert_eq!(
            logical_error.identifier(),
            Some("RunMat:compatibility:CovLogicalDataExtension")
        );

        let data = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let normalization_error = block_on(cov_builtin(
            Value::Tensor(data.clone()),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .unwrap_err();
        assert_eq!(
            normalization_error.identifier(),
            Some("RunMat:compatibility:CovTypedNormalizationExtension")
        );

        let weights = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let weight_error = block_on(cov_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap()),
            vec![Value::Tensor(weights)],
        ))
        .unwrap_err();
        assert_eq!(
            weight_error.identifier(),
            Some("RunMat:compatibility:CovVectorWeightsExtension")
        );
    }

    #[test]
    fn cov_supports_all_eight_integer_classes_and_wide_centering() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ];
        for storage in storages {
            let class = storage.class_name();
            let data = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
            let Value::Tensor(covariance) = block_on(cov_builtin(Value::Tensor(data), Vec::new()))
                .unwrap_or_else(|error| panic!("{class}: {error}"))
            else {
                panic!("{class}: expected tensor");
            };
            assert_tensor_close(&covariance, &[1.0], 1.0e-12);

            let left = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let right = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
            let Value::Tensor(pair) =
                block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
                    .unwrap_or_else(|error| panic!("{class} paired data: {error}"))
            else {
                panic!("{class}: expected paired tensor");
            };
            assert_tensor_close(&pair, &[1.0; 4], 1.0e-12);

            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap();
            let weights = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let weighted = block_on(cov_builtin(
                Value::Tensor(matrix),
                vec![Value::Tensor(weights)],
            ))
            .unwrap_or_else(|error| panic!("{class} weights: {error}"));
            assert!(matches!(weighted, Value::Tensor(_)));
        }

        let base = 1_u64 << 63;
        let wide = Tensor::new_integer(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            vec![3, 1],
        )
        .unwrap();
        let Value::Tensor(covariance) =
            block_on(cov_builtin(Value::Tensor(wide), Vec::new())).expect("wide covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&covariance, &[1.0], 1.0e-12);
    }

    #[test]
    fn cov_resident_integer_data_rejects_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                block_on(cov_builtin(Value::GpuTensor(handle.clone()), Vec::new())).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:CovIntegerDataExtension")
            );
            drop(handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_omitrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                3.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                4.0,
                6.0,
                9.0, //
                2.0,
                5.0,
                7.0,
                10.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitrows")],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_partialrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                4.0,
                7.0, //
                2.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                6.0,
                9.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("partialrows")],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            9.0,
            18.0,
            4.5, //
            18.0,
            18.0,
            f64::NAN, //
            4.5,
            f64::NAN,
            4.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_mismatched_rows_errors() {
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
            .expect_err("expected mismatch error");
        assert_eq!(err.identifier(), COV_ERROR_ROWS_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_invalid_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(Value::Tensor(tensor), vec![Value::Num(2.5)]))
            .expect_err("expected invalid flag error");
        assert_eq!(err.identifier(), COV_ERROR_NORMALIZATION_INVALID.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weight_vector_length_mismatch_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = Tensor::new(vec![10.0, 11.0, 12.0], vec![3, 1]).unwrap();
        let w = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(y), Value::Tensor(w)],
        ))
        .expect_err("expected weight length mismatch");
        assert_eq!(
            err.identifier(),
            COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_unknown_rows_option_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("bogus")],
        ))
        .expect_err("expected unknown rows option error");
        assert_eq!(err.identifier(), COV_ERROR_ROWS_OPTION_UNKNOWN.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_duplicate_normalization_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .expect_err("expected duplicate normalization flag error");
        assert_eq!(
            err.identifier(),
            COV_ERROR_NORMALIZATION_DUPLICATE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_too_many_array_arguments_errors() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        let w = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let z = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(y), Value::Tensor(w), Value::Tensor(z)],
        ))
        .expect_err("expected too many array arguments error");
        assert_eq!(
            err.identifier(),
            COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(cov_builtin(Value::GpuTensor(handle), Vec::new())).expect("cov");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.0250, 0.0075, //
                0.0075, 0.0070,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_host_weights_return_resident_result() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();

            let result = block_on(cov_builtin(
                Value::GpuTensor(handle),
                vec![Value::Tensor(weights)],
            ))
            .expect("weighted cov");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.022380952380952376,
                0.004999999999999994, //
                0.004999999999999994,
                0.006666666666666678,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_resident_weights_return_resident_result() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();
            let data = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload data");
            let weight_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &weights.materialize_f64(),
                    shape: &weights.shape,
                })
                .expect("upload weights");

            let result = block_on(cov_builtin(
                Value::GpuTensor(data),
                vec![Value::GpuTensor(weight_handle)],
            ))
            .expect("weighted cov");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.022380952380952376,
                0.004999999999999994, //
                0.004999999999999994,
                0.006666666666666678,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_rejects_negative_resident_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let weights = Tensor::new(vec![1.0, -1.0], vec![2, 1]).unwrap();
            let data = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload data");
            let weight_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &weights.materialize_f64(),
                    shape: &weights.shape,
                })
                .expect("upload weights");

            let err = block_on(cov_builtin(
                Value::GpuTensor(data),
                vec![Value::GpuTensor(weight_handle)],
            ))
            .expect_err("negative weights should fail");
            assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();

        let cpu_result =
            block_on(cov_builtin(Value::Tensor(tensor.clone()), Vec::new())).expect("cov");
        let cpu_tensor = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("cov");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.materialize_f64(), 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_paired_matrices_vectorize_to_two_variables() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let left_handle = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
        let right_handle = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
        let result = cov_builtin_sync(
            Value::GpuTensor(left_handle),
            vec![Value::GpuTensor(right_handle)],
        )
        .expect("paired covariance");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather");
        assert_tensor_close(&gathered, &[5.0 / 3.0; 4], 1.0e-5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_row_scalar_and_empty_match_host_shapes() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        for (data, shape, expected) in [
            (vec![1.0, 2.0, 3.0], vec![1, 3], 1.0),
            (vec![7.0], Vec::new(), 0.0),
            (Vec::new(), vec![0, 0], f64::NAN),
        ] {
            let tensor = Tensor::new(data, shape).expect("tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("covariance");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_tensor_close(&gathered, &[expected], 1.0e-6);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_weighted_matches_cpu_and_stays_resident() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();

        let cpu_result = block_on(cov_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(weights.clone())],
        ))
        .expect("cov");
        let cpu_tensor = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let data_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            })
            .expect("upload data");
        let weight_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &weights.materialize_f64(),
                shape: &weights.shape,
            })
            .expect("upload weights");

        let gpu_value = cov_builtin_sync(
            Value::GpuTensor(data_handle),
            vec![Value::GpuTensor(weight_handle)],
        )
        .expect("weighted cov");
        assert!(matches!(gpu_value, Value::GpuTensor(_)));
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.materialize_f64(), 1.0e-5);
    }
}
