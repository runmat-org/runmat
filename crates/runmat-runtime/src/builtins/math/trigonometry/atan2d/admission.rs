//! Pure operand/shape admission before any provider download or publication.
use super::*;

struct Operand<'a> {
    shape: &'a [usize],
}

fn operand(value: &Value) -> BuiltinResult<Operand<'_>> {
    super::super::inverse_helpers::ensure_input_extensions(
        value,
        NAME,
        &ATAN2D_INTEGER_INPUT_EXTENSION,
        &ATAN2D_LOGICAL_INPUT_EXTENSION,
        &ATAN2D_CHARACTER_INPUT_EXTENSION,
    )?;
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1][..],
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => &array.shape,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(error(&ATAN2D_ERROR_COMPLEX_UNSUPPORTED, "complex input"));
        }
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err()
                || !gpu_helpers::expected_handle_numeric_element_type(handle).is_ok_and(|element| {
                    gpu_helpers::numeric_descriptor_matches(
                        handle,
                        element,
                        runmat_accelerate_api::handle_storage(handle),
                    )
                })
            {
                return Err(error(
                    &ATAN2D_ERROR_INTERNAL,
                    "malformed or unowned GPU input",
                ));
            }
            if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
                return Err(error(&ATAN2D_ERROR_COMPLEX_UNSUPPORTED, "complex input"));
            }
            handle.shape()
        }
        Value::CharArray(_) => unreachable!("character shape is borrowed in pair"),
        _ => {
            return Err(error(
                &ATAN2D_ERROR_INVALID_INPUT,
                "expected real numeric input",
            ))
        }
    };
    Ok(Operand { shape })
}

pub(super) fn pair(y_value: &Value, x_value: &Value) -> BuiltinResult<()> {
    // Character arrays keep their own row/column cache, as in radian atan2.
    let y_chars;
    let x_chars;
    let y = if let Value::CharArray(chars) = y_value {
        super::super::inverse_helpers::ensure_input_extensions(
            y_value,
            NAME,
            &ATAN2D_INTEGER_INPUT_EXTENSION,
            &ATAN2D_LOGICAL_INPUT_EXTENSION,
            &ATAN2D_CHARACTER_INPUT_EXTENSION,
        )?;
        y_chars = [chars.rows, chars.cols];
        Operand { shape: &y_chars }
    } else {
        operand(y_value)?
    };
    let x = if let Value::CharArray(chars) = x_value {
        super::super::inverse_helpers::ensure_input_extensions(
            x_value,
            NAME,
            &ATAN2D_INTEGER_INPUT_EXTENSION,
            &ATAN2D_LOGICAL_INPUT_EXTENSION,
            &ATAN2D_CHARACTER_INPUT_EXTENSION,
        )?;
        x_chars = [chars.rows, chars.cols];
        Operand { shape: &x_chars }
    } else {
        operand(x_value)?
    };
    BroadcastPlan::new(y.shape, x.shape)
        .map_err(|cause| error(&ATAN2D_ERROR_SIZE_MISMATCH, cause))?;
    gpu_helpers::require_resident_compute_values(&[y_value, x_value], true, || {
        error(
            &ATAN2D_ERROR_INVALID_INPUT,
            "resident inputs require one exact owning provider",
        )
    })?;
    if let (Value::GpuTensor(y), Value::GpuTensor(x)) = (y_value, x_value) {
        gpu_helpers::exact_provider_for_binary_inputs(y, x)
            .map_err(|cause| error(&ATAN2D_ERROR_INVALID_INPUT, cause))?;
    }
    Ok(())
}

pub(super) fn host_integer_exactness(y: &Value, x: &Value) -> BuiltinResult<()> {
    for input in [y, x] {
        if matches!(input, Value::Int(_))
            || matches!(input, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        {
            super::super::inverse_helpers::ensure_integer_exact_f64(input, NAME)?;
        }
    }
    Ok(())
}

pub(super) async fn exact_integer_boundaries(y: &Value, x: &Value) -> BuiltinResult<()> {
    for input in [y, x] {
        if !matches!(input, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        {
            continue;
        }
        crate::builtins::common::validation::ensure_runmat_real_integer_f64_boundary(
            input,
            &ATAN2D_INTEGER_INPUT_EXTENSION,
            runmat_builtins::ATAN2D_CATALOG_ENTRY.identity,
            &ATAN2D_ERROR_INVALID_INPUT,
            "Y or X",
        )
        .await?;
    }
    Ok(())
}
