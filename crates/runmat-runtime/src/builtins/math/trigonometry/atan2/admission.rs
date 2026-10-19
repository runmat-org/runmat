//! Validate both coordinate operands before placement or provider access.
use super::*;

struct Operand<'a> {
    shape: &'a [usize],
    floating: bool,
}

fn operand(value: &Value) -> BuiltinResult<Operand<'_>> {
    ensure_atan2_input_extensions(value)?;
    let (shape, floating): (&[usize], bool) = match value {
        Value::Num(_) => (&[1, 1], true),
        Value::Int(_) | Value::Bool(_) => (&[1, 1], false),
        Value::Tensor(tensor) => (
            tensor.shape(),
            matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ),
        ),
        Value::LogicalArray(array) => (&array.shape, false),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(atan2_error(&ATAN2_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::GpuTensor(handle) => {
            if gpu_helpers::exact_provider_for_handle(handle).is_none()
                || !gpu_helpers::expected_handle_numeric_element_type(handle).is_ok_and(|element| {
                    gpu_helpers::numeric_descriptor_matches(
                        handle,
                        element,
                        runmat_accelerate_api::handle_storage(handle),
                    )
                })
            {
                return Err(atan2_error_with_detail(
                    &ATAN2_ERROR_INTERNAL,
                    "malformed or unowned GPU input",
                ));
            }
            if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
                return Err(atan2_error(&ATAN2_ERROR_COMPLEX_UNSUPPORTED));
            }
            (
                handle.shape(),
                runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle),
            )
        }
        _ => return Err(atan2_error(&ATAN2_ERROR_INVALID_INPUT)),
    };
    Ok(Operand { shape, floating })
}

pub(super) fn pair(y: &Value, x: &Value) -> BuiltinResult<bool> {
    // Character dimensions are represented by rows/cols rather than a borrowed
    // shape slice. Keep their existing column-major conversion in the host body.
    let y_chars;
    let x_chars;
    let y = if let Value::CharArray(chars) = y {
        ensure_atan2_input_extensions(y)?;
        y_chars = [chars.rows, chars.cols];
        Operand {
            shape: &y_chars,
            floating: false,
        }
    } else {
        operand(y)?
    };
    let x = if let Value::CharArray(chars) = x {
        ensure_atan2_input_extensions(x)?;
        x_chars = [chars.rows, chars.cols];
        Operand {
            shape: &x_chars,
            floating: false,
        }
    } else {
        operand(x)?
    };
    BroadcastPlan::new(y.shape, x.shape)
        .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_SIZE_MISMATCH, error))?;
    Ok(y.floating && x.floating)
}
