use super::*;

#[derive(Clone, serde::Serialize)]
pub(super) struct RangeSpec {
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) end_row: Option<usize>,
    pub(super) end_col: Option<usize>,
}

impl RangeSpec {
    pub(super) fn validate(&self) -> BuiltinResult<()> {
        if let Some(end_row) = self.end_row {
            if end_row < self.start_row {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_RANGE,
                    "readmatrix: Range end row must be >= start row",
                ));
            }
        }
        if let Some(end_col) = self.end_col {
            if end_col < self.start_col {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_RANGE,
                    "readmatrix: Range end column must be >= start column",
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn normalize_missing_token(token: &str) -> String {
    token.trim().to_ascii_lowercase()
}

pub(super) fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => parse_range_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                readmatrix_error_with(
                    &READMATRIX_ERROR_RANGE,
                    "readmatrix: Range contains malformed UTF-16",
                )
            })?;
            parse_range_string(&text)
        }
        Value::StringArray(sa) => {
            if sa.data().len() == 1 && !sa.data()[0].is_missing() {
                parse_range_string(
                    sa.data()[0]
                        .as_str()
                        .expect("checked nonmissing scalar string"),
                )
            } else {
                Err(readmatrix_error_with(
                    &READMATRIX_ERROR_RANGE,
                    "readmatrix: Range string array inputs must be scalar",
                ))
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range must be provided as a string or numeric vector",
        )),
    }
}

fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range string cannot be empty",
        ));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            format!("readmatrix: invalid Range specification '{}'", text),
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range must specify a starting column",
        ));
    }
    let end_ref = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end) = end_ref {
        if end.col.is_none() {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                "readmatrix: Range end must include a column reference",
            ));
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end_ref.as_ref().and_then(|r| r.row);
    let end_col = end_ref.as_ref().and_then(|r| r.col);
    Ok(RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

pub(super) fn parse_range_numeric(value: &Value) -> BuiltinResult<RangeSpec> {
    let elements = match value {
        Value::Tensor(t) => tensor::tensor_values_f64(t),
        _ => {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                "readmatrix: numeric Range must be provided as a vector with 2 or 4 elements",
            ))
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: numeric Range must contain exactly 2 or 4 elements",
        ));
    }
    let mut indices = Vec::with_capacity(elements.len());
    for (idx, value) in elements.iter().enumerate() {
        let converted = positive_index(*value, idx)?;
        indices.push(converted);
    }
    let start_row = indices[0];
    let start_col = indices[1];
    let (end_row, end_col) = if indices.len() == 4 {
        (Some(indices[2]), Some(indices[3]))
    } else {
        (None, None)
    };
    Ok(RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

fn positive_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range indices must be finite",
        ));
    }
    if value < 1.0 {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range indices must be >= 1",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range indices must be integers",
        ));
    }
    let zero_based = (rounded as i64) - 1;
    if zero_based < 0 {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range indices must be >= 1",
        ));
    }
    usize::try_from(zero_based).map_err(|_| {
        readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            format!(
                "readmatrix: Range index {} is too large to fit in usize",
                position + 1
            ),
        )
    })
}

#[derive(Clone, Copy)]
struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

fn parse_cell_reference(token: &str) -> BuiltinResult<CellReference> {
    let mut letters = String::new();
    let mut digits = String::new();
    for ch in token.trim().chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                format!("readmatrix: invalid Range component '{}'", token),
            ));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range references cannot be empty",
        ));
    }
    let col = if letters.is_empty() {
        None
    } else {
        Some(column_index_from_letters(&letters)?)
    };
    let row = if digits.is_empty() {
        None
    } else {
        let parsed = digits.parse::<usize>().map_err(|_| {
            readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                format!(
                    "readmatrix: invalid row index '{}' in Range component '{}'",
                    digits, token
                ),
            )
        })?;
        if parsed == 0 {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                "readmatrix: Range rows must be >= 1",
            ));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_RANGE,
                format!(
                    "readmatrix: invalid column designator '{}' in Range",
                    letters
                ),
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| {
                readmatrix_error_with(
                    &READMATRIX_ERROR_RANGE,
                    "readmatrix: Range column index overflowed",
                )
            })?;
    }
    value.checked_sub(1).ok_or_else(|| {
        readmatrix_error_with(
            &READMATRIX_ERROR_RANGE,
            "readmatrix: Range column index underflowed",
        )
    })
}

pub(super) fn apply_range<T: Clone>(
    rows: Vec<Vec<T>>,
    max_cols: usize,
    range: &RangeSpec,
    default_fill: T,
) -> (Vec<Vec<T>>, usize) {
    if rows.is_empty() || max_cols == 0 {
        return (Vec::new(), 0);
    }
    if range.start_row >= rows.len() || range.start_col >= max_cols {
        return (Vec::new(), 0);
    }
    let last_row = rows.len().saturating_sub(1);
    let mut end_row = range.end_row.unwrap_or(last_row);
    if end_row > last_row {
        end_row = last_row;
    }
    if end_row < range.start_row {
        return (Vec::new(), 0);
    }

    let last_col = max_cols.saturating_sub(1);
    let mut end_col = range.end_col.unwrap_or(last_col);
    if end_col > last_col {
        end_col = last_col;
    }
    if end_col < range.start_col {
        return (Vec::new(), 0);
    }

    let mut subset = Vec::new();
    let mut subset_max_cols = 0usize;
    for row in rows
        .into_iter()
        .skip(range.start_row)
        .take(end_row - range.start_row + 1)
    {
        let mut extracted = Vec::with_capacity(end_col - range.start_col + 1);
        let mut fields = row.into_iter().skip(range.start_col);
        for _ in range.start_col..=end_col {
            let value = fields.next().unwrap_or_else(|| default_fill.clone());
            extracted.push(value);
        }
        subset_max_cols = subset_max_cols.max(extracted.len());
        subset.push(extracted);
    }

    if subset_max_cols == 0 {
        (Vec::new(), 0)
    } else {
        (subset, subset_max_cols)
    }
}
