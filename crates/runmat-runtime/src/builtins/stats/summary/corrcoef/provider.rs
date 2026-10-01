use super::*;

pub(super) async fn corrcoef_try_gpu(args: &CorrcoefArgs) -> BuiltinResult<Option<Value>> {
    if args.rows != CorrcoefRows::All {
        return Ok(None);
    }
    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };
    if first_handle.shape().len() > 2 {
        return Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_INVALID_ARGUMENT,
            "A must be a matrix",
        ));
    }
    if runmat_accelerate_api::handle_integer_type(first_handle).is_some()
        || runmat_accelerate_api::handle_is_logical(first_handle)
        || runmat_accelerate_api::handle_storage(first_handle)
            == GpuTensorStorage::ComplexInterleaved
    {
        return Ok(None);
    }
    let provider = match gpu_helpers::exact_provider_for_handle(first_handle) {
        Some(provider) => provider,
        None => return Ok(None),
    };
    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => {
            if !same_size(first_handle.shape(), handle.shape()) {
                return Err(corrcoef_error(&CORRCOEF_ERROR_SIZE_MISMATCH));
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    == GpuTensorStorage::ComplexInterleaved
            {
                return Ok(None);
            }
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

    let _first_residency =
        crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(first_handle);
    let _second_residency =
        maybe_second_handle.map(crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new);
    if maybe_second_handle.is_some_and(|second| {
        runmat_accelerate_api::handle_precision(first_handle)
            != runmat_accelerate_api::handle_precision(second)
    }) {
        // Concatenation requires a common physical class. The admitted host
        // path promotes mixed observation classes before the calculation.
        return Ok(None);
    }
    let elements = first_handle
        .shape()
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            corrcoef_error_with_detail(&CORRCOEF_ERROR_INVALID_ARGUMENT, "input shape overflow")
        })?;
    if maybe_second_handle.is_some() && elements == 1 {
        // Paired scalars are one two-observation variable, not two one-row
        // variables. The generic provider hook receives columnwise inputs.
        return Ok(None);
    }
    let matrix_handle = if let Some(second) = maybe_second_handle {
        let left = match provider.reshape(first_handle, &[elements, 1]) {
            Ok(handle) => handle,
            Err(error)
                if crate::builtins::common::gpu_helpers::provider_hook_is_unsupported(&error) =>
            {
                return Ok(None)
            }
            Err(error) => return Err(corrcoef_provider_error(error.into())),
        };
        let right = match provider.reshape(second, &[elements, 1]) {
            Ok(handle) => handle,
            Err(error)
                if crate::builtins::common::gpu_helpers::provider_hook_is_unsupported(&error) =>
            {
                return Ok(None)
            }
            Err(error) => return Err(corrcoef_provider_error(error.into())),
        };
        match provider.cat(2, &[left, right]) {
            Ok(concat) => concat,
            Err(error)
                if crate::builtins::common::gpu_helpers::provider_hook_is_unsupported(&error) =>
            {
                return Ok(None)
            }
            Err(error) => return Err(corrcoef_provider_error(error.into())),
        }
    } else {
        first_handle.clone()
    };

    let options = CorrcoefOptions {
        normalization: CorrcoefNormalization::Unbiased,
        rows: args.rows,
    };

    match provider.corrcoef(&matrix_handle, &options).await {
        Ok(result) => {
            let precision = CovariancePrecision::observations(&args.first, args.second.as_ref());
            let Some(mut result) = precision
                .align_provider(provider, result)
                .await
                .map_err(|error| corrcoef_provider_error(error.into()))?
            else {
                return Ok(None);
            };
            let output_width = if maybe_second_handle.is_some() {
                if elements == 1 {
                    1
                } else {
                    2
                }
            } else {
                match first_handle.shape() {
                    [] | [_] => 1,
                    [rows, columns] if *rows == 1 || *columns == 1 => 1,
                    [_, columns] => *columns,
                    _ => unreachable!("preflight rejected rank above two"),
                }
            };
            let protected: Vec<_> = std::iter::once(first_handle)
                .chain(maybe_second_handle)
                .collect();
            let valid = gpu_helpers::unary_gpu_output_matches_shape(
                &result,
                first_handle,
                provider,
                &[output_width, output_width],
                gpu_helpers::UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: Some(precision.provider_precision()),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
                },
            ) && protected
                .iter()
                .all(|input| !result.shares_allocation_with(input));
            if !valid {
                let mut error =
                    corrcoef_internal_error("corrcoef: invalid provider correlation output");
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            if let Err(error) = result.allocation_lease().validate_live() {
                let error = corrcoef_provider_error(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                );
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            gpu_helpers::propagate_output_provenance(&mut result, protected).map_err(|error| {
                corrcoef_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                    error,
                ))
            })?;
            gpu_helpers::resident_gpu_value(result)
                .map(Some)
                .map_err(|error| {
                    corrcoef_provider_error(
                        runmat_accelerate_api::ProviderExecutionError::from_source(error),
                    )
                })
        }
        Err(error) => {
            if error.is_unsupported() {
                Ok(None)
            } else {
                Err(corrcoef_provider_error(error))
            }
        }
    }
}
