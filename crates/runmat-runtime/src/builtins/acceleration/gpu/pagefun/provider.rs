use super::*;
pub(super) fn try_pagefun_gpu(
    operation: &PageOperation,
    operands: &[Value],
) -> BuiltinResult<Option<Value>> {
    if operands.is_empty() {
        return Ok(None);
    }
    if !operands
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    if operands.iter().any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some() || runmat_accelerate_api::handle_is_logical(handle))
    }) {
        return Ok(None);
    }

    let handles: Vec<GpuTensorHandle> = operands
        .iter()
        .map(|value| match value {
            Value::GpuTensor(handle) => handle.clone(),
            _ => unreachable!(),
        })
        .collect();
    let provider = gpu_helpers::exact_provider_for_binary_inputs(&handles[0], &handles[1])
        .map_err(pagefun_internal_error)?;
    if handles.iter().any(|handle| {
        runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_precision(handle) != Some(provider.precision())
    }) {
        return Ok(None);
    }

    let request = match build_pagefun_request(operation, &handles)? {
        Some(request) => request,
        None => return Ok(None),
    };

    match provider.pagefun(&request) {
        Ok(handle) => {
            let contract = gpu_helpers::BinaryGpuOutputContract {
                shape: request.output_shape,
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
                precision: Some(provider.precision()),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
            };
            if !gpu_helpers::binary_gpu_output_matches(
                &handle,
                &handles[0],
                &handles[1],
                provider,
                &contract,
            ) {
                return Err(gpu_helpers::reject_gpu_output(
                    handle,
                    pagefun_internal_error(
                        "provider returned invalid page-product metadata or input alias",
                    ),
                ));
            }
            let handle = gpu_helpers::propagate_owned_output_provenance(handle, handles.iter())?;
            Ok(Some(gpu_helpers::resident_gpu_value(handle)?))
        }
        Err(err) if gpu_helpers::provider_hook_allows_host_fallback(&err) => Ok(None),
        Err(err) => Err(pagefun_internal_error(format!(
            "provider page product failed: {err}"
        ))),
    }
}

pub(super) fn build_pagefun_request(
    operation: &PageOperation,
    handles: &[GpuTensorHandle],
) -> BuiltinResult<Option<PagefunRequest>> {
    match operation {
        PageOperation::Mtimes => {
            if handles.len() != 2 {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_INVALID_ARITY,
                    "@mtimes requires exactly two array inputs",
                ));
            }

            let (lhs_rows, lhs_cols, lhs_pages) = handle_matrix_meta(&handles[0])?;
            let (rhs_rows, rhs_cols, rhs_pages) = handle_matrix_meta(&handles[1])?;
            // The page kernel is a matrix product, not scalar expansion.
            if lhs_rows * lhs_cols == 1 || rhs_rows * rhs_cols == 1 {
                return Ok(None);
            }
            if lhs_cols != rhs_rows {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_MATRIX_DIM_MISMATCH,
                    format!(
                        "inner matrix dimensions must agree ({}x{} * {}x{})",
                        lhs_rows, lhs_cols, rhs_rows, rhs_cols
                    ),
                ));
            }

            let rank = lhs_pages.len().max(rhs_pages.len());
            let result_page_dims = broadcast_page_dims(&lhs_pages, &rhs_pages)?;

            let mut input_page_dims = Vec::with_capacity(2);
            let mut lhs_padded = lhs_pages.clone();
            lhs_padded.resize(rank, 1);
            let mut rhs_padded = rhs_pages.clone();
            rhs_padded.resize(rank, 1);
            input_page_dims.push(lhs_padded);
            input_page_dims.push(rhs_padded);

            let mut output_shape = vec![lhs_rows, rhs_cols];
            output_shape.extend_from_slice(&result_page_dims);
            checked_volume(&output_shape)?;

            Ok(Some(PagefunRequest {
                op: PagefunOp::Mtimes,
                inputs: handles.to_vec(),
                output_shape,
                page_dims: result_page_dims,
                input_page_dims,
            }))
        }
    }
}

pub(super) fn handle_matrix_meta(
    handle: &GpuTensorHandle,
) -> BuiltinResult<(usize, usize, Vec<usize>)> {
    let canonical = canonical_matrix_shape(handle.shape());
    if canonical.len() < 2 {
        return Err(pagefun_error_with_detail(
            &PAGEFUN_ERROR_INVALID_INPUT,
            "gpu tensor must be at least 2-D",
        ));
    }
    let rows = canonical[0];
    let cols = canonical[1];
    let pages = if canonical.len() > 2 {
        canonical[2..].to_vec()
    } else {
        Vec::new()
    };
    Ok((rows, cols, pages))
}
