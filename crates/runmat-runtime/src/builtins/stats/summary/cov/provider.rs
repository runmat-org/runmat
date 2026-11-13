use super::*;

pub(super) async fn cov_try_gpu(args: &CovArgs) -> BuiltinResult<Option<Value>> {
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
    if maybe_second_handle.is_some_and(|second| {
        runmat_accelerate_api::handle_precision(first_handle)
            != runmat_accelerate_api::handle_precision(second)
    }) {
        // Use the admitted host promotion path for mixed observation classes.
        return Ok(None);
    }

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
    let precision = CovariancePrecision::observations(&args.first, args.second.as_ref());
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

    match provider
        .covariance(
            first_handle,
            maybe_second_handle,
            weight_handle.as_ref(),
            &options,
        )
        .await
    {
        Ok(result) => {
            let result = precision
                .align_provider(
                    provider,
                    result,
                    CovarianceAlignmentContract {
                        expected_shape: &[columns, columns],
                        protected: &borrowed,
                        builtin: NAME,
                    },
                    |reason| cov_internal_error(format!("cov: {reason}")),
                    cov_provider_error,
                )
                .await?;
            match result {
                AlignedCovarianceOutput::Resident(result) => {
                    let result = gpu_helpers::propagate_owned_output_provenance(result, borrowed)
                        .map_err(|error| {
                        cov_provider_error(
                            runmat_accelerate_api::ProviderExecutionError::from_source(error),
                        )
                    })?;
                    gpu_helpers::resident_gpu_value(result)
                        .map(Some)
                        .map_err(|error| {
                            cov_provider_error(
                                runmat_accelerate_api::ProviderExecutionError::from_source(error),
                            )
                        })
                }
                AlignedCovarianceOutput::Materialized(value) => {
                    gpu_helpers::restore_class_preserving_value_protecting(
                        first_handle,
                        value,
                        NAME,
                        &borrowed,
                    )
                    .map(Some)
                    .map_err(|mut error| {
                        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        error
                    })
                }
            }
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

pub(super) fn covariance_observation_count(
    shape: &[usize],
    paired_input: bool,
) -> BuiltinResult<usize> {
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
            let weights = super::host::value_to_weight_vector(other.clone(), expected_rows).await?;
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
                    if error.is_unsupported() =>
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

pub(super) fn validate_weight_shape(shape: &[usize], expected_rows: usize) -> BuiltinResult<()> {
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
