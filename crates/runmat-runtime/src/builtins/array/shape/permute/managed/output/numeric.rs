use super::*;

pub(in crate::builtins::array::shape::permute) fn tensor(
    context: &RuntimeContext,
    builtin: &'static str,
    source: &Tensor,
    order: &[usize],
) -> BuiltinResult<Option<Tensor>> {
    if !managed_policy(context, builtin)? {
        return Ok(None);
    }
    let dtype = source.numeric_dtype();
    let retained =
        ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, source.len(), order.len())
            .map_err(numeric_error)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
    let Some(prepared) = PreparedPermutation::prepare(
        context,
        builtin,
        source.shape(),
        source.len(),
        order,
        retained,
        ExactHostTensorAllocation::retained_claim_count(),
    )?
    else {
        return Ok(None);
    };
    let mut output = prepared
        .receipt
        .exact_output_tensor(context, dtype, prepared.output_shape())
        .map_err(|cause| allocation_error(context, builtin, cause))?;
    macro_rules! copy_integer {
        ($variant:ident, $method:ident) => {
            if let Some(runmat_value::IntegerStorage::$variant(source)) = source.integer_storage() {
                copy_native(
                    context,
                    builtin,
                    &prepared,
                    source,
                    output.$method().expect("exact output class"),
                )?;
            } else {
                return Err(permute_error(
                    builtin,
                    format!("{builtin}: exact integer source class changed"),
                ));
            }
        };
    }
    match dtype {
        NumericDType::F64 => copy_native(
            context,
            builtin,
            &prepared,
            source.as_f64_slice().expect("validated double storage"),
            output.as_f64_slice_mut().expect("exact double output"),
        )?,
        NumericDType::F32 => copy_native(
            context,
            builtin,
            &prepared,
            source.as_f32_slice().expect("validated single storage"),
            output.as_f32_slice_mut().expect("exact single output"),
        )?,
        NumericDType::I8 => copy_integer!(I8, as_i8_slice_mut),
        NumericDType::I16 => copy_integer!(I16, as_i16_slice_mut),
        NumericDType::I32 => copy_integer!(I32, as_i32_slice_mut),
        NumericDType::I64 => copy_integer!(I64, as_i64_slice_mut),
        NumericDType::U8 => copy_integer!(U8, as_u8_slice_mut),
        NumericDType::U16 => copy_integer!(U16, as_u16_slice_mut),
        NumericDType::U32 => copy_integer!(U32, as_u32_slice_mut),
        NumericDType::U64 => copy_integer!(U64, as_u64_slice_mut),
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

fn copy_native<T: Copy>(
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
