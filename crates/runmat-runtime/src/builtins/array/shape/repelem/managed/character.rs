use super::*;

pub(super) fn assemble(
    context: &RuntimeContext,
    source: &CharArray,
    controls: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    let source_len = source
        .rows
        .checked_mul(source.cols)
        .ok_or_else(|| repelem_invalid_factors("repelem: character source dimensions overflow"))?;
    if source_len != source.data().len() {
        return Err(repelem_internal(
            "repelem: character source shape does not match payload",
        ));
    }
    let matrix_shape = match source.shape() {
        [] => Some((0, 0)),
        [length] => Some((1, *length)),
        [rows, cols, rest @ ..] if rest.iter().all(|&dim| dim == 1) => Some((*rows, *cols)),
        _ => None,
    };
    if matrix_shape != Some((source.rows, source.cols)) {
        return Err(repelem_internal(
            "repelem: character shape cache does not match dimensions",
        ));
    }
    if controls.len() > 2 {
        for (index, control) in controls.iter().enumerate().skip(2) {
            if control_len(control, index + 1)? != 1 || control_at(control, 0, index + 1)? != 1 {
                return Err(repelem_invalid_factors(format!(
                    "repelem: char and cell arrays only support replication along the first two dimensions (extra factor at position {} must be 1)",
                    index + 1
                )));
            }
        }
    }
    let single_axis = if controls.len() == 1 {
        Some(vector_replication_axis(&[source.rows, source.cols])?)
    } else {
        None
    };
    let rows = axis_extent(context, controls, single_axis, 0, source.rows)?;
    let cols = axis_extent(context, controls, single_axis, 1, source.cols)?;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| repelem_invalid_factors("repelem: requested output exceeds maximum size"))?;
    let table_len = rows
        .checked_add(cols)
        .ok_or_else(|| repelem_invalid_factors("repelem: metadata extent overflow"))?;
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(table_len)
        .map_err(|cause| admission(context, cause))?;
    let output = crate::builtins::common::character_output::build(
        context,
        2,
        count,
        scratch,
        |axis| Ok(if axis == 0 { rows } else { cols }),
        |data, _, receipt| {
            let mut table = receipt
                .temporary_typed::<usize>(context, table_len)
                .map_err(|cause| admission(context, cause))?;
            let (row_table, col_table) = table.as_mut_slice().split_at_mut(rows);
            fill_axis_table(
                context,
                receipt,
                axis_control(controls, single_axis, 0),
                source.rows,
                row_table,
            )?;
            fill_axis_table(
                context,
                receipt,
                axis_control(controls, single_axis, 1),
                source.cols,
                col_table,
            )?;
            for row in 0..rows {
                for col in 0..cols {
                    let destination = row * cols + col;
                    if destination % CHECKPOINT_CHUNK == 0 {
                        receipt
                            .checkpoint(context)
                            .map_err(|cause| admission(context, cause))?;
                    }
                    let index = row_table[row] * source.cols + col_table[col];
                    data[destination] = *source.data().get(index).ok_or_else(|| {
                        repelem_internal(
                            "repelem: character source geometry changed during assembly",
                        )
                    })?;
                }
            }
            Ok(())
        },
        crate::builtins::common::character_output::CharacterBuildErrors {
            admission,
            invalid: |detail| repelem_internal(format!("repelem: {detail}")),
        },
    )?;
    Ok(Value::CharArray(output))
}
