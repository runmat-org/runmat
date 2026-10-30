use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct RangeSpec {
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) end_row: Option<usize>,
    pub(super) end_col: Option<usize>,
}

impl RangeSpec {
    pub(super) fn relative_to_loaded_rows(self) -> Self {
        if self.end_row.is_some_and(|end_row| end_row < self.start_row) {
            return Self {
                start_row: 1,
                start_col: self.start_col,
                end_row: Some(0),
                end_col: self.end_col,
            };
        }
        Self {
            start_row: 0,
            start_col: self.start_col,
            end_row: self.end_row.map(|end_row| end_row - self.start_row),
            end_col: self.end_col,
        }
    }
}

pub(super) fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => parse_range_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                csvread_error_with(
                    &CSVREAD_ERROR_RANGE,
                    "csvread: Range contains malformed UTF-16",
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
                Err(csvread_error_with(
                    &CSVREAD_ERROR_RANGE,
                    "csvread: Range string array inputs must be scalar",
                ))
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range must be provided as a string or numeric vector",
        )),
    }
}

pub(super) fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range string cannot be empty",
        ));
    }
    let (separator, extension) = if trimmed.contains("..") {
        ("..", None)
    } else if trimmed.contains(':') {
        (":", Some(&CSVREAD_COLON_RANGE_EXTENSION))
    } else {
        ("..", None)
    };
    if let Some(extension) = extension {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    let parts: Vec<&str> = trimmed.split(separator).collect();
    if parts.len() > 2 {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            format!("csvread: invalid Range specification '{trimmed}'"),
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range must specify a starting column",
        ));
    }
    let end = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end_ref) = end {
        if end_ref.col.is_none() {
            return Err(csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                "csvread: Range end must include a column reference",
            ));
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end.as_ref().and_then(|r| r.row);
    let end_col = end.as_ref().and_then(|r| r.col);
    Ok(RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

pub(super) fn parse_range_numeric(value: &Value) -> BuiltinResult<RangeSpec> {
    let elements = match value {
        Value::Tensor(t) => (0..tensor::tensor_element_len(t))
            .map(|idx| {
                t.numeric_value_at(idx)
                    .expect("index within authoritative numeric storage")
            })
            .collect::<Vec<_>>(),
        _ => {
            return Err(csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                "csvread: numeric Range must be provided as a vector with 2 or 4 elements",
            ));
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: numeric Range must contain exactly 2 or 4 elements",
        ));
    }
    if elements.len() == 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVREAD_TWO_VECTOR_RANGE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut indices = Vec::with_capacity(elements.len());
    for (idx, element) in elements.into_iter().enumerate() {
        if let Some(integer) = element.into_int_value() {
            indices.push(integer.try_to_usize().ok_or_else(|| {
                csvread_error_with(
                    &CSVREAD_ERROR_RANGE,
                    format!(
                        "csvread: Range index {} must be a representable non-negative integer",
                        idx + 1
                    ),
                )
            })?);
        } else {
            indices.push(non_negative_index(element.materialize_f64(), idx)?);
        }
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

fn non_negative_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range indices must be finite",
        ));
    }
    if value < 0.0 {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range indices must be non-negative",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range indices must be integers",
        ));
    }
    usize::try_from(rounded as i64).map_err(|_| {
        csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            format!(
                "csvread: Range index {} is too large to fit in usize",
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
            return Err(csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                format!("csvread: invalid Range component '{token}'"),
            ));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range references cannot be empty",
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
            csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                format!(
                    "csvread: invalid row index '{}' in Range component '{token}'",
                    digits
                ),
            )
        })?;
        if parsed == 0 {
            return Err(csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                "csvread: Range rows must be >= 1",
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
            return Err(csvread_error_with(
                &CSVREAD_ERROR_RANGE,
                format!("csvread: invalid column designator '{letters}' in Range"),
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| {
                csvread_error_with(
                    &CSVREAD_ERROR_RANGE,
                    "csvread: Range column index overflowed",
                )
            })?;
    }
    value.checked_sub(1).ok_or_else(|| {
        csvread_error_with(
            &CSVREAD_ERROR_RANGE,
            "csvread: Range column index underflowed",
        )
    })
}
