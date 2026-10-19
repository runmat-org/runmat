use super::super::scan;
use super::*;
use std::io::{Seek, SeekFrom};

mod storage;
use storage::FillStorage;

fn selected_geometry(
    geometry: &BoundedScanGeometry,
    range: Option<&RangeSpec>,
) -> BuiltinResult<(usize, usize, usize, usize)> {
    if geometry.rows == 0 || geometry.columns == 0 {
        return Ok((0, 0, 0, 0));
    }
    let Some(range) = range else {
        return Ok((0, 0, geometry.rows, geometry.columns));
    };
    if range.start_row >= geometry.rows || range.start_col >= geometry.columns {
        return Ok((0, 0, 0, 0));
    }
    let last_row = range
        .end_row
        .unwrap_or(geometry.rows - 1)
        .min(geometry.rows - 1);
    let last_col = range
        .end_col
        .unwrap_or(geometry.columns - 1)
        .min(geometry.columns - 1);
    if last_row < range.start_row || last_col < range.start_col {
        return Ok((0, 0, 0, 0));
    }
    Ok((
        range.start_row,
        range.start_col,
        last_row - range.start_row + 1,
        last_col - range.start_col + 1,
    ))
}

pub(in crate::builtins::io::tabular::readmatrix::reader) async fn fill_utf8(
    source: &mut FsBoundedInput,
    geometry: &BoundedScanGeometry,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    options: &ReadMatrixOptions,
    mut resources: FillResources<'_>,
) -> BuiltinResult<Tensor> {
    let (first_row, first_col, rows, cols) = selected_geometry(geometry, options.range.as_ref())?;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| matrix_overflow_with(resources.diagnostic))?;
    source.seek(SeekFrom::Start(0)).map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: unable to replay the retained input source",
            source,
        )
    })?;
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let integer_dtype = integer_output_dtype(&options.output_template);
    let mut storage =
        FillStorage::new(rows, cols, count, integer_dtype, options, resources.output)?;
    let mut reader = RowReader::new(source, resources.line.take(), resources.diagnostic);
    let mut data_row = 0usize;
    let mut semantic_error = None;
    while let Some(line_number) = reader
        .next(cooperation, path, options.num_header_lines)
        .await?
    {
        let selected =
            semantic_error.is_none() && data_row >= first_row && data_row - first_row < rows;
        let mut target = FillTarget {
            first_col,
            rows,
            integer_dtype,
            options,
            storage: &mut storage,
            diagnostic_reservation: resources.diagnostic,
        };
        let field_count = scan::for_each_field_ref(
            reader.text(),
            &geometry.delimiter,
            cooperation,
            path,
            resources.field.as_mut(),
            |source_col, field| {
                if selected
                    && semantic_error.is_none()
                    && source_col >= first_col
                    && source_col - first_col < cols
                {
                    if let Err(error) = target.write(
                        field.as_str(),
                        source_col,
                        data_row - first_row,
                        line_number,
                    ) {
                        semantic_error = Some(error);
                    }
                }
                Ok(())
            },
        )
        .await?;
        if field_count == 0 {
            continue;
        }
        if field_count > geometry.columns || data_row >= geometry.rows {
            return Err(changed_source(resources.diagnostic));
        }
        if selected && semantic_error.is_none() {
            let last_col = first_col + cols;
            let mut padded = 0;
            for source_col in field_count.max(first_col)..last_col {
                if let Err(error) = target.write("", source_col, data_row - first_row, line_number)
                {
                    semantic_error = Some(error);
                    break;
                }
                padded += 1;
                if padded >= ROW_WORK_CHUNK {
                    checked_row_work(cooperation, path, padded).await?;
                    padded = 0;
                }
            }
            if padded != 0 {
                checked_row_work(cooperation, path, padded).await?;
            }
        }
        data_row = data_row
            .checked_add(1)
            .ok_or_else(|| matrix_overflow_with(resources.diagnostic))?;
    }
    if data_row != geometry.rows {
        return Err(changed_source(resources.diagnostic));
    }
    let replay_digest = reader.finish_digest();
    if replay_digest != geometry.source_digest {
        return Err(changed_source(resources.diagnostic));
    }
    if let Some(error) = semantic_error {
        return Err(error);
    }
    storage.into_tensor(rows, cols)
}

fn changed_source(reservation: Option<&dyn DiagnosticReservation>) -> RuntimeError {
    FixedReaderDiagnostic::ChangedSource.render(reservation)
}

struct FillTarget<'a> {
    first_col: usize,
    rows: usize,
    integer_dtype: Option<NumericDType>,
    options: &'a ReadMatrixOptions,
    storage: &'a mut FillStorage,
    diagnostic_reservation: Option<&'a dyn DiagnosticReservation>,
}

impl FillTarget<'_> {
    fn write(
        &mut self,
        token: &str,
        source_col: usize,
        output_row: usize,
        line_number: usize,
    ) -> BuiltinResult<()> {
        let output_col = source_col - self.first_col;
        let index = output_col * self.rows + output_row;
        if let Some(dtype) = self.integer_dtype {
            let value = parse_integer_numeric_token_with_reservation(
                token,
                self.options,
                dtype,
                line_number,
                self.options.column_from_source(source_col),
                self.diagnostic_reservation,
            )?;
            self.storage.set(index, value)?;
        } else {
            let value = parse_numeric_token_with_reservation(
                token,
                self.options,
                line_number,
                self.options.column_from_source(source_col),
                self.diagnostic_reservation,
            )?;
            self.storage.set(index, NumericScalar::F64(value))?;
        }
        Ok(())
    }
}
