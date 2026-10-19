use super::*;

mod numeric;
pub(in crate::builtins::array::shape::permute) use numeric::tensor;

pub(in crate::builtins::array::shape::permute) fn logical(
    context: &RuntimeContext,
    builtin: &'static str,
    source: &LogicalArray,
    order: &[usize],
) -> BuiltinResult<Option<LogicalArray>> {
    if !managed_policy(context, builtin)? {
        return Ok(None);
    }
    let retained =
        ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(source.len(), order.len())
            .map_err(numeric_error)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
    let Some(prepared) = PreparedPermutation::prepare(
        context,
        builtin,
        &source.shape,
        source.len(),
        order,
        retained,
        ExactHostLogicalTensorAllocation::retained_claim_count(),
    )?
    else {
        return Ok(None);
    };
    let mut output = prepared
        .receipt
        .with_output_allocation(context, |permit| {
            ExactHostLogicalTensorAllocation::new(prepared.output_shape(), permit)
        })
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    for destination in 0..prepared.len {
        if destination.is_multiple_of(CHECKPOINT_CHUNK) {
            prepared
                .receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        let source_index = prepared.source_index(destination)?;
        output.as_mut_slice()[destination] = source.data[source_index];
    }
    prepared
        .receipt
        .checkpoint(context)
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    Ok(Some(output.into_array().map_err(numeric_error).map_err(
        |cause| allocation_error(context, builtin, cause),
    )?))
}

pub(in crate::builtins::array::shape::permute) fn complex(
    context: &RuntimeContext,
    builtin: &'static str,
    source: &ComplexTensor,
    order: &[usize],
) -> BuiltinResult<Option<ComplexTensor>> {
    if !managed_policy(context, builtin)? {
        return Ok(None);
    }
    if matches!(source.complex_storage(), ComplexStorage::Integer(_)) {
        return integer_complex(context, builtin, source, order);
    }
    let dtype = source.numeric_dtype();
    let retained = ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(
        dtype,
        source.len(),
        order.len(),
    )
    .map_err(numeric_error)
    .map_err(|cause| allocation_error(context, builtin, cause))?;
    let Some(prepared) = PreparedPermutation::prepare(
        context,
        builtin,
        source.shape(),
        source.len(),
        order,
        retained,
        ExactHostComplexTensorAllocation::retained_claim_count(),
    )?
    else {
        return Ok(None);
    };
    let output = prepared
        .receipt
        .with_output_allocation(context, |permit| {
            ExactHostComplexTensorAllocation::new(dtype, prepared.output_shape(), permit)
        })
        .map_err(|cause| allocation_error(context, builtin, cause))?
        .into_tensor()
        .map_err(numeric_error)
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    let mut output = output;
    match dtype {
        NumericDType::F32 => {
            let source_values = source.as_f32_slice().ok_or_else(|| {
                permute_error(
                    builtin,
                    format!("{builtin}: expected native single complex input"),
                )
            })?;
            let output_values = output.try_as_f32_slice_mut_unique().ok_or_else(|| {
                permute_error(
                    builtin,
                    format!("{builtin}: admitted complex output is not unique"),
                )
            })?;
            copy_complex(context, builtin, &prepared, source_values, output_values)?;
        }
        NumericDType::F64 => {
            let source_values = source.as_f64_slice().ok_or_else(|| {
                permute_error(
                    builtin,
                    format!("{builtin}: expected native double complex input"),
                )
            })?;
            let output_values = output.try_as_f64_slice_mut_unique().ok_or_else(|| {
                permute_error(
                    builtin,
                    format!("{builtin}: admitted complex output is not unique"),
                )
            })?;
            copy_complex(context, builtin, &prepared, source_values, output_values)?;
        }
        _ => unreachable!("paired integer complex rejected before output preparation"),
    }
    prepared
        .receipt
        .checkpoint(context)
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    Ok(Some(output))
}

fn integer_complex(
    context: &RuntimeContext,
    builtin: &'static str,
    source: &ComplexTensor,
    order: &[usize],
) -> BuiltinResult<Option<ComplexTensor>> {
    let dtype = source.numeric_dtype();
    let retained = ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
        dtype,
        source.len(),
        order.len(),
    )
    .map_err(numeric_error)
    .map_err(|cause| allocation_error(context, builtin, cause))?;
    let Some(prepared) = PreparedPermutation::prepare(
        context,
        builtin,
        source.shape(),
        source.len(),
        order,
        retained,
        ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
    )?
    else {
        return Ok(None);
    };
    let mut output = prepared
        .receipt
        .with_output_allocation(context, |permit| {
            ExactHostIntegerComplexTensorAllocation::new(dtype, prepared.output_shape(), permit)
        })
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    for destination in 0..prepared.len {
        if destination.is_multiple_of(CHECKPOINT_CHUNK) {
            prepared
                .receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        let source_index = prepared.source_index(destination)?;
        let (real, imag) = source.numeric_value_at(source_index).ok_or_else(|| {
            permute_error(builtin, format!("{builtin}: invalid complex source index"))
        })?;
        output
            .set(destination, real, imag)
            .map_err(|error| permute_error(builtin, format!("{builtin}: {error}")))?;
    }
    prepared
        .receipt
        .checkpoint(context)
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    output
        .into_tensor()
        .map(Some)
        .map_err(numeric_error)
        .map_err(|cause| allocation_error(context, builtin, cause))
}

fn copy_complex<T: Copy>(
    context: &RuntimeContext,
    builtin: &'static str,
    prepared: &PreparedPermutation,
    source: &[T],
    output: &mut [T],
) -> BuiltinResult<()> {
    for destination in 0..prepared.len {
        if destination.is_multiple_of(CHECKPOINT_CHUNK) {
            prepared
                .receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        output[destination] = source[prepared.source_index(destination)?];
    }
    Ok(())
}
