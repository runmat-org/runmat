use super::*;

pub(super) struct SubsetResult {
    pub(super) rows: Vec<Vec<ParsedNumber>>,
    pub(super) row_count: usize,
    pub(super) col_count: usize,
}

pub(super) async fn apply_offsets(
    cooperation: &mut TabularCooperation<'_>,
    rows: &[Vec<ParsedNumber>],
    max_cols: usize,
    start_row: usize,
    start_col: usize,
    default_fill: ParsedNumber,
) -> BuiltinResult<SubsetResult> {
    if rows.is_empty() || max_cols == 0 {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }
    if start_row >= rows.len() {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }
    if start_col >= max_cols {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }

    let mut subset_rows = Vec::new();
    let mut col_count = 0usize;
    for row in rows.iter().skip(start_row) {
        if start_col >= row.len() && row.len() < max_cols {
            // Entire row missing required columns; fill zeros of remaining width.
            let width = max_cols - start_col;
            let mut extracted = Vec::with_capacity(width);
            for _ in 0..width {
                extracted.push(default_fill);
                if extracted.len() % 1024 == 0 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(extracted.len() % 1024).await?;
            subset_rows.push(extracted);
            col_count = col_count.max(width);
            continue;
        }
        let mut extracted = Vec::with_capacity(max_cols - start_col);
        for col_idx in start_col..max_cols {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            extracted.push(value);
            if extracted.len() % 1024 == 0 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(extracted.len() % 1024).await?;
        col_count = col_count.max(extracted.len());
        subset_rows.push(extracted);
    }
    let row_count = subset_rows.len();
    Ok(SubsetResult {
        rows: subset_rows,
        row_count,
        col_count,
    })
}

pub(super) async fn apply_range(
    cooperation: &mut TabularCooperation<'_>,
    rows: &[Vec<ParsedNumber>],
    max_cols: usize,
    range: &RangeSpec,
    default_fill: ParsedNumber,
) -> BuiltinResult<SubsetResult> {
    if rows.is_empty() || max_cols == 0 {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }
    if range.start_row >= rows.len() || range.start_col >= max_cols {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }
    let last_row = rows.len().saturating_sub(1);
    let mut end_row = range.end_row.unwrap_or(last_row);
    if end_row > last_row {
        end_row = last_row;
    }
    if end_row < range.start_row {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }

    let last_col = max_cols.saturating_sub(1);
    let mut end_col = range.end_col.unwrap_or(last_col);
    if end_col > last_col {
        end_col = last_col;
    }
    if end_col < range.start_col {
        return Ok(SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        });
    }

    let mut subset_rows = Vec::new();
    let mut col_count = 0usize;
    for row_idx in range.start_row..=end_row {
        if row_idx >= rows.len() {
            break;
        }
        let row = &rows[row_idx];
        let mut extracted = Vec::with_capacity(end_col - range.start_col + 1);
        for col_idx in range.start_col..=end_col {
            if col_idx >= max_cols {
                break;
            }
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            extracted.push(value);
            if extracted.len() % 1024 == 0 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(extracted.len() % 1024).await?;
        col_count = col_count.max(extracted.len());
        subset_rows.push(extracted);
    }
    let row_count = subset_rows.len();
    Ok(SubsetResult {
        rows: subset_rows,
        row_count,
        col_count,
    })
}

pub(super) async fn rows_to_value(
    cooperation: &mut TabularCooperation<'_>,
    rows: Vec<Vec<ParsedNumber>>,
    row_count: usize,
    col_count: usize,
    default_fill: ParsedNumber,
) -> BuiltinResult<Value> {
    if row_count == 0 || col_count == 0 {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|e| csvread_error_with(&CSVREAD_ERROR_TENSOR_BUILD, format!("csvread: {e}")));
    }
    let element_count = row_count.checked_mul(col_count).ok_or_else(|| {
        csvread_error_with(&CSVREAD_ERROR_TENSOR_BUILD, "csvread: output size overflow")
    })?;
    let mut data = Vec::with_capacity(element_count);
    for _ in 0..element_count {
        data.push(default_fill);
        if data.len() % 1024 == 0 {
            cooperation.completed(1024).await?;
        }
    }
    cooperation.completed(data.len() % 1024).await?;
    for (row_idx, row) in rows.iter().enumerate().take(row_count) {
        for col_idx in 0..col_count {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            data[row_idx + col_idx * row_count] = value;
            if col_idx % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(col_count % 1024).await?;
    }
    let mut complex = false;
    for chunk in data.chunks(1024) {
        complex |= chunk.iter().any(|value| value.complex);
        cooperation.completed(chunk.len()).await?;
    }
    if complex {
        let mut encoded = Vec::with_capacity(data.len());
        for chunk in data.chunks(1024) {
            encoded.extend(chunk.iter().map(|value| (value.re, value.im)));
            cooperation.completed(chunk.len()).await?;
        }
        ComplexTensor::new(encoded, vec![row_count, col_count])
            .map(Value::ComplexTensor)
            .map_err(|e| csvread_error_with(&CSVREAD_ERROR_TENSOR_BUILD, format!("csvread: {e}")))
    } else {
        let mut encoded = Vec::with_capacity(data.len());
        for chunk in data.chunks(1024) {
            encoded.extend(chunk.iter().map(|value| value.re));
            cooperation.completed(chunk.len()).await?;
        }
        Tensor::new(encoded, vec![row_count, col_count])
            .map(Value::Tensor)
            .map_err(|e| csvread_error_with(&CSVREAD_ERROR_TENSOR_BUILD, format!("csvread: {e}")))
    }
}
