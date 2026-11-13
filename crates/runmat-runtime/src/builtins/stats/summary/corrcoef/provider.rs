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
        // These are metadata-only views. A provider reshape may allocate, so
        // it cannot precede a replayable optional concatenation attempt.
        if first_handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
            || second.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
        {
            return Ok(None);
        }
        let left = first_handle
            .with_view(first_handle.view().clone().with_shape(vec![elements, 1]))
            .map_err(|error| {
                corrcoef_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                    error,
                ))
            })?;
        let right = second
            .with_view(second.view().clone().with_shape(vec![elements, 1]))
            .map_err(|error| {
                corrcoef_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                    error,
                ))
            })?;
        match provider.cat_optional(2, &[left, right]).await {
            Ok(mut concat) => {
                let same_owner =
                    gpu_helpers::exact_provider_for_handle(&concat).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, provider)
                    });
                if !same_owner
                    || concat.shape() != [elements, 2]
                    || concat.physical().element_type() != first_handle.physical().element_type()
                    || concat.physical().storage() != first_handle.physical().storage()
                    || concat.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                    || concat.shares_allocation_with(first_handle)
                    || concat.shares_allocation_with(second)
                {
                    return Err(gpu_helpers::reject_gpu_output(
                        concat,
                        corrcoef_internal_error(
                            "corrcoef: concatenation returned a foreign or input-aliasing output",
                        ),
                    ));
                }
                if let Err(error) =
                    gpu_helpers::propagate_output_provenance(&mut concat, [first_handle, second])
                {
                    return Err(gpu_helpers::reject_gpu_output(concat, error));
                }
                concat
            }
            Err(error) if error.is_unsupported() => return Ok(None),
            Err(error) => return Err(corrcoef_provider_error(error)),
        }
    } else {
        first_handle.clone()
    };

    let options = CorrcoefOptions {
        normalization: CorrcoefNormalization::Unbiased,
        rows: args.rows,
    };
    let precision = CovariancePrecision::observations(&args.first, args.second.as_ref());
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
    let borrowed: Vec<_> = std::iter::once(first_handle)
        .chain(maybe_second_handle)
        .collect();
    let protected: Vec<_> = borrowed
        .iter()
        .copied()
        .chain(std::iter::once(&matrix_handle))
        .collect();

    let outcome: BuiltinResult<Option<Value>> = async {
        match provider.corrcoef(&matrix_handle, &options).await {
            Ok(result) => {
                let result = precision
                    .align_provider(
                        provider,
                        result,
                        CovarianceAlignmentContract {
                            expected_shape: &[output_width, output_width],
                            protected: &protected,
                            builtin: NAME,
                        },
                        |reason| corrcoef_internal_error(format!("corrcoef: {reason}")),
                        corrcoef_provider_error,
                    )
                    .await?;
                match result {
                    AlignedCovarianceOutput::Resident(mut result) => {
                        if let Err(error) = gpu_helpers::propagate_output_provenance(
                            &mut result,
                            borrowed.iter().copied(),
                        ) {
                            return Err(gpu_helpers::reject_gpu_output(result, error));
                        }
                        gpu_helpers::resident_gpu_value(result)
                            .map(Some)
                            .map_err(|error| {
                                corrcoef_provider_error(
                                    runmat_accelerate_api::ProviderExecutionError::from_source(
                                        error,
                                    ),
                                )
                            })
                    }
                    AlignedCovarianceOutput::Materialized(value) => Ok(Some(value)),
                }
            }
            Err(error) => {
                if error.is_unsupported() && maybe_second_handle.is_none() {
                    Ok(None)
                } else {
                    Err(corrcoef_provider_error(if maybe_second_handle.is_some() {
                        error.after_effect()
                    } else {
                        error
                    }))
                }
            }
        }
    }
    .await;
    if maybe_second_handle.is_some() {
        if let Err(error) = outcome {
            return Err(gpu_helpers::reject_gpu_output(matrix_handle, error));
        }
        if let Err(cleanup) = matrix_handle.release() {
            let error = corrcoef_internal_error(format!(
                "corrcoef: concatenated observation retirement failed: {cleanup}"
            ));
            return Err(match outcome {
                Ok(Some(Value::GpuTensor(output))) => gpu_helpers::reject_gpu_output(output, error),
                _ => error,
            });
        }
        return outcome;
    }
    outcome
}
