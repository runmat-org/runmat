use super::*;

pub(super) fn checked_volume(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
        .ok_or_else(|| {
            pagefun_error_with_detail(
                &PAGEFUN_ERROR_INVALID_INPUT,
                "array shape exceeds the supported size",
            )
        })
}

pub(super) fn broadcast_page_dims(left: &[usize], right: &[usize]) -> BuiltinResult<Vec<usize>> {
    (0..left.len().max(right.len()))
        .map(|axis| {
            let l = left.get(axis).copied().unwrap_or(1);
            let r = right.get(axis).copied().unwrap_or(1);
            if l == r || r == 1 {
                Ok(l)
            } else if l == 1 {
                Ok(r)
            } else {
                Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_PAGE_DIM_MISMATCH,
                    format!("page dimension {} mismatch ({l} vs {r})", axis + 3),
                ))
            }
        })
        .collect()
}

pub(super) fn validate_operand_metadata(operands: &[Value]) -> BuiltinResult<()> {
    let mut shapes = Vec::with_capacity(operands.len());
    for value in operands {
        if crate::builtins::common::validation::is_typed_complex_integer(value) {
            return Err(pagefun_error_with_detail(
                &PAGEFUN_ERROR_INVALID_INPUT,
                "complex integer page arithmetic is not supported",
            ));
        }
        let shape = match value {
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(..) => vec![1, 1],
            Value::Tensor(t) => canonical_matrix_shape(t.shape()),
            Value::ComplexTensor(t) => canonical_matrix_shape(t.shape()),
            Value::LogicalArray(t) => canonical_matrix_shape(&t.shape),
            Value::GpuTensor(handle) => canonical_matrix_shape(handle.shape()),
            _ => return Err(pagefun_error(&PAGEFUN_ERROR_INVALID_INPUT)),
        };
        checked_volume(&shape)?;
        checked_volume(&shape[..2])?;
        shapes.push(shape);
    }
    let left = &shapes[0];
    let right = &shapes[1];
    let left_scalar = left[0] == 1 && left[1] == 1;
    let right_scalar = right[0] == 1 && right[1] == 1;
    crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs(
        &operands[0],
        &operands[1],
        left_scalar,
        right_scalar,
        BUILTIN_NAME,
    )
    .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_INVALID_INPUT, error))?;
    if !left_scalar && !right_scalar && left[1] != right[0] {
        return Err(pagefun_error(&PAGEFUN_ERROR_MATRIX_DIM_MISMATCH));
    }
    let pages = broadcast_page_dims(&left[2..], &right[2..])?;
    let (rows, cols) = if left_scalar {
        (right[0], right[1])
    } else if right_scalar {
        (left[0], left[1])
    } else {
        (left[0], right[1])
    };
    checked_volume(&assemble_shape(rows, cols, &pages))?;
    let is_integer = |value: &Value| match value {
        Value::Int(_) => true,
        Value::Tensor(t) => t.integer_storage().is_some(),
        Value::GpuTensor(h) => runmat_accelerate_api::handle_integer_type(h).is_some(),
        _ => false,
    };
    if operands.iter().any(is_integer) && !left_scalar && !right_scalar {
        return Err(pagefun_error_with_detail(
            &PAGEFUN_ERROR_INVALID_INPUT,
            "integer page multiplication requires a scalar page operand",
        ));
    }
    Ok(())
}

pub(super) fn canonical_matrix_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => {
            let mut out = shape.to_vec();
            if out.len() == 1 {
                out.push(1);
            }
            out
        }
    }
}
