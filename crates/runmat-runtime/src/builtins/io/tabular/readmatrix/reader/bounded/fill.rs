use super::super::scan;
use super::*;
use std::io::{Seek, SeekFrom};

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
    path: &TabularFsPath,
    options: &ReadMatrixOptions,
    staged_line: Option<ExactHostEffectBytes>,
) -> BuiltinResult<Tensor> {
    let (first_row, first_col, rows, cols) = selected_geometry(geometry, options.range.as_ref())?;
    let count = rows.checked_mul(cols).ok_or_else(matrix_overflow)?;
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
    let mut integer = integer_dtype.map(|dtype| NumericStorage::zeros(dtype, count));
    let mut floating = integer_dtype
        .is_none()
        .then(|| vec![options.empty_value_f64(); count]);
    let mut reader = RowReader::new(source, staged_line);
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
            integer: &mut integer,
            floating: &mut floating,
        };
        let field_count = scan::for_each_field(
            reader.text(),
            &geometry.delimiter,
            cooperation,
            path,
            |source_col, field| {
                if selected
                    && semantic_error.is_none()
                    && source_col >= first_col
                    && source_col - first_col < cols
                {
                    if let Err(error) = target.write(
                        field.as_ref(),
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
            return Err(changed_source());
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
        data_row = data_row.checked_add(1).ok_or_else(matrix_overflow)?;
    }
    if data_row != geometry.rows {
        return Err(changed_source());
    }
    let replay_digest = reader.finish_digest();
    if replay_digest != geometry.source_digest {
        return Err(changed_source());
    }
    if let Some(error) = semantic_error {
        return Err(error);
    }
    if rows == 0 || cols == 0 {
        return empty_tensor();
    }
    if let Some(storage) = integer {
        Tensor::from_numeric_storage(storage, vec![rows, cols]).map_err(|source| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {source}"),
            )
        })
    } else {
        Tensor::new(floating.expect("floating storage"), vec![rows, cols]).map_err(|source| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {source}"),
            )
        })
    }
}

fn changed_source() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_IO_READ,
        "readmatrix: opened source changed between validation and output",
    )
}

struct FillTarget<'a> {
    first_col: usize,
    rows: usize,
    integer_dtype: Option<NumericDType>,
    options: &'a ReadMatrixOptions,
    integer: &'a mut Option<NumericStorage>,
    floating: &'a mut Option<Vec<f64>>,
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
        if let (Some(dtype), Some(storage)) = (self.integer_dtype, self.integer.as_mut()) {
            let value = parse_integer_numeric_token(
                token,
                self.options,
                dtype,
                line_number,
                source_col + 1,
            )?;
            storage.set_value(index, value).map_err(|source| {
                readmatrix_error_with(
                    &READMATRIX_ERROR_TENSOR_BUILD,
                    format!("readmatrix: {source}"),
                )
            })?;
        } else if let Some(storage) = self.floating.as_mut() {
            storage[index] = parse_numeric_token(token, self.options, line_number, source_col + 1)?;
        }
        Ok(())
    }
}
