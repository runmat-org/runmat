//! Diagonal metadata shares the concrete parser, dimensions and public modes.
use super::*;
pub(in crate::builtins) struct DiagonalGeometry {
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
    output_columns: usize,
    start_row: usize,
    start_column: usize,
    vector: bool,
    passthrough: bool,
}
impl DiagonalGeometry {
    pub(in crate::builtins) fn copies_uniform(&self) -> bool {
        !self.vector || self.passthrough
    }
    pub(in crate::builtins) fn output_shape(&self) -> [usize; 2] {
        [self.output_rows, self.output_columns]
    }
    pub(in crate::builtins) fn source(&self, output: usize) -> Option<usize> {
        if self.passthrough {
            return (output < self.output_rows).then_some(output);
        }
        if self.vector {
            if self.output_rows == 0 {
                return None;
            }
            let row = output % self.output_rows;
            let column = output / self.output_rows;
            if column >= self.output_columns || row < self.start_row || column < self.start_column {
                return None;
            }
            let row = row - self.start_row;
            let column = column - self.start_column;
            let count = self.input_rows.checked_mul(self.input_columns)?;
            (row == column && row < count).then_some(row)
        } else {
            if output >= self.output_rows {
                return None;
            }
            Some(self.start_row + output + (self.start_column + output) * self.input_rows)
        }
    }
    pub(in crate::builtins) fn project(&self, input: usize) -> Option<usize> {
        if self.passthrough {
            return Some(input);
        }
        if self.vector {
            let row = self.start_row.checked_add(input)?;
            let column = self.start_column.checked_add(input)?;
            if row < self.output_rows && column < self.output_columns {
                Some(row + column * self.output_rows)
            } else {
                None
            }
        } else {
            if self.input_rows == 0 {
                return None;
            }
            let row = input % self.input_rows;
            let column = input / self.input_rows;
            if column >= self.input_columns || row < self.start_row || column < self.start_column {
                return None;
            }
            let row = row - self.start_row;
            let column = column - self.start_column;
            (row == column && row < self.output_rows).then_some(row)
        }
    }
}
pub(in crate::builtins) async fn diagonal_geometry(
    shape: &[usize],
    arguments: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<DiagonalGeometry>> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        "diag affine geometry",
        "complete metadata planning workspace is not certified",
    )?;
    crate::context::RuntimeScopeFactory::new(runtime.cloned(), move || async move {
        ensure_diag_geometry_compatibility(Some(shape), arguments)?;
        let args = ParsedDiagArgs::parse(arguments).await?;
        if !matches!(
            args.template,
            OutputTemplate::Native | OutputTemplate::Double
        ) {
            return Ok(None);
        }
        let (rows, columns) = matrix_dims(shape)?;
        let vector = rows == 1 || columns == 1;
        validate_vector_mode(vector, &args)?;
        let shift = args.offset.unsigned_abs();
        let (start_row, start_column) = if args.offset >= 0 {
            (0, shift)
        } else {
            (shift, 0)
        };
        let (output_rows, output_columns) = if args.vector_mode {
            (vector_length(rows, columns), 1)
        } else if vector {
            vector_output_dims(vector_length(rows, columns), &args)?
        } else {
            if args.size_override.is_some() {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_INPUT,
                    "diag: size overrides require vector inputs",
                ));
            }
            (
                if start_row >= rows || start_column >= columns {
                    0
                } else {
                    (rows - start_row).min(columns - start_column)
                },
                1,
            )
        };
        Ok(Some(DiagonalGeometry {
            input_rows: rows,
            input_columns: columns,
            output_rows,
            output_columns,
            start_row,
            start_column,
            vector,
            passthrough: args.vector_mode,
        }))
    })
    .construct()
    .await
}
