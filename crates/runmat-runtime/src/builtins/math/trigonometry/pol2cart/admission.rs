use super::*;

/// Validate the entire form before any downloads, uploads, or provider hooks.
pub(super) fn admit(inputs: &[Value], outputs: Option<usize>) -> BuiltinResult<Vec<usize>> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(pol2cart_error(
            &ERROR_INVALID_INPUT,
            "expected two or three inputs",
        ));
    }
    if outputs.is_some_and(|count| count > inputs.len()) {
        return Err(pol2cart_error(
            &ERROR_OUTPUT_COUNT,
            "requested outputs exceed the selected syntax",
        ));
    }
    let mut shape = vec![1, 1];
    for value in inputs {
        let next = match value {
            Value::Num(_) | Value::Int(_) => vec![1, 1],
            Value::Tensor(tensor) => tensor.shape().to_vec(),
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
                    return Err(pol2cart_error(
                        &ERROR_INTERNAL,
                        "GPU input has no owning provider",
                    ));
                }
                if handle.descriptor().storage == Some(GpuTensorStorage::ComplexInterleaved) {
                    return Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"));
                }
                let element = gpu_helpers::expected_handle_numeric_element_type(handle)
                    .map_err(|error| pol2cart_error(&ERROR_INVALID_INPUT, error))?;
                if !gpu_helpers::numeric_descriptor_matches(handle, element, GpuTensorStorage::Real)
                    || runmat_accelerate_api::handle_is_logical(handle)
                {
                    return Err(pol2cart_error(
                        &ERROR_INVALID_INPUT,
                        "expected explicit real numeric GPU storage",
                    ));
                }
                handle.shape().to_vec()
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                return Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"));
            }
            _ => {
                return Err(pol2cart_error(
                    &ERROR_INVALID_INPUT,
                    "expected real dense numeric input",
                ))
            }
        };
        shape = matlab_broadcast_shape(&shape, &next)?;
    }
    if !shape.contains(&0)
        && shape
            .iter()
            .try_fold(1usize, |n, size| n.checked_mul(*size))
            .is_none()
    {
        return Err(pol2cart_error(
            &ERROR_SIZE_MISMATCH,
            "output element count overflows",
        ));
    }
    // Independently valid owners may use the established mixed-owner host
    // formula. Validate every source without inventing a same-owner contract.
    gpu_helpers::require_resident_compute_value_iter(inputs.iter(), false, || {
        pol2cart_error(&ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    // Gate every integer role before any role's exactness check may download.
    for (value, extension) in inputs.iter().zip([
        &POL2CART_INTEGER_THETA_EXTENSION,
        &POL2CART_INTEGER_RHO_EXTENSION,
        &POL2CART_INTEGER_Z_EXTENSION,
    ]) {
        if matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
            || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
        {
            crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
        }
        super::super::inverse_helpers::ensure_integer_exact_f64(value, NAME)?;
    }
    Ok(shape)
}
