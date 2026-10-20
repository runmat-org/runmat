use super::*;

pub(super) fn assemble(
    context: &RuntimeContext,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let (rows, cols) = values
        .iter()
        .try_fold((0usize, 0usize), |(rows, cols), value| {
            let Value::CharArray(chars) = value else {
                return Err(error_with_detail(
                    &BLKDIAG_ERROR_INVALID_INPUT,
                    "character block changed class",
                ));
            };
            let expected = chars
                .rows
                .checked_mul(chars.cols)
                .ok_or_else(size_overflow)?;
            let matrix_shape = match chars.shape() {
                [] => Some((0, 0)),
                [length] => Some((1, *length)),
                [rows, cols, rest @ ..] if rest.iter().all(|&dim| dim == 1) => Some((*rows, *cols)),
                _ => None,
            };
            if chars.data().len() != expected || matrix_shape != Some((chars.rows, chars.cols)) {
                return Err(error_with_detail(
                    &BLKDIAG_ERROR_INVALID_INPUT,
                    "character block shape does not match storage",
                ));
            }
            Ok((
                rows.checked_add(chars.rows).ok_or_else(size_overflow)?,
                cols.checked_add(chars.cols).ok_or_else(size_overflow)?,
            ))
        })?;
    let count = checked_len(rows, cols)?;
    let output = crate::builtins::common::character_output::build(
        context,
        2,
        count,
        0,
        |axis| Ok(if axis == 0 { rows } else { cols }),
        |data, _, receipt| {
            let mut row_offset = 0usize;
            let mut col_offset = 0usize;
            for value in values {
                let Value::CharArray(chars) = value else {
                    return Err(error_with_detail(
                        &BLKDIAG_ERROR_INVALID_INPUT,
                        "character block changed class",
                    ));
                };
                for row in 0..chars.rows {
                    for col in 0..chars.cols {
                        let dst = (row_offset + row) * cols + col_offset + col;
                        if dst.is_multiple_of(CHECKPOINT_CHUNK) {
                            receipt
                                .checkpoint(context)
                                .map_err(|cause| admission(context, cause))?;
                        }
                        data[dst] = chars.data()[row * chars.cols + col];
                    }
                }
                row_offset += chars.rows;
                col_offset += chars.cols;
            }
            Ok(())
        },
        crate::builtins::common::character_output::CharacterBuildErrors {
            admission,
            invalid: |detail| error_with_detail(&BLKDIAG_ERROR_INVALID_INPUT, detail),
        },
    )?;
    Ok(Value::CharArray(output))
}
