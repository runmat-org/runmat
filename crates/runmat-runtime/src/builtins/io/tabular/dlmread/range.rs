use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct RangeSpec {
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) end_row: Option<usize>,
    pub(super) end_col: Option<usize>,
}

pub(super) fn validate_range(spec: RangeSpec) -> BuiltinResult<RangeSpec> {
    if let Some(end_row) = spec.end_row {
        if end_row < spec.start_row {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                "dlmread: Range must satisfy R1 <= R2 and C1 <= C2",
            ));
        }
    }
    if let Some(end_col) = spec.end_col {
        if end_col < spec.start_col {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                "dlmread: Range must satisfy R1 <= R2 and C1 <= C2",
            ));
        }
    }
    Ok(spec)
}

pub(super) fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => parse_range_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                dlmread_error_with(
                    &DLMREAD_ERROR_RANGE,
                    "dlmread: Range contains malformed UTF-16",
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
                Err(dlmread_error_with(
                    &DLMREAD_ERROR_RANGE,
                    "dlmread: Range string array inputs must be scalar",
                ))
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range must be provided as a string or numeric vector",
        )),
    }
}

pub(super) fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range string cannot be empty",
        ));
    }
    let parts: Vec<&str> = if trimmed.contains("..") {
        trimmed.split("..").collect()
    } else {
        if trimmed.contains(':') {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DLMREAD_COLON_RANGE_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        trimmed.split(':').collect()
    };
    if parts.len() > 2 {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            format!("dlmread: invalid Range specification '{trimmed}'"),
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range must specify a starting column",
        ));
    }
    let end = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end_ref) = end {
        if end_ref.col.is_none() {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                "dlmread: Range end must include a column reference",
            ));
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end.as_ref().and_then(|r| r.row);
    let end_col = end.as_ref().and_then(|r| r.col);
    let spec = RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    };
    validate_range(spec)
}

pub(super) fn parse_range_numeric(value: &Value) -> BuiltinResult<RangeSpec> {
    let elements = match value {
        Value::Tensor(t) => (0..t.len())
            .map(|index| {
                t.numeric_value_at(index).ok_or_else(|| {
                    dlmread_error_with(
                        &DLMREAD_ERROR_RANGE,
                        "dlmread: numeric Range contains an invalid element",
                    )
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?,
        _ => {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                "dlmread: numeric Range must be provided as a vector with 2 or 4 elements",
            ))
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: numeric Range must contain exactly 2 or 4 elements",
        ));
    }
    let mut indices = Vec::with_capacity(elements.len());
    for (idx, element) in elements.iter().enumerate() {
        indices.push(non_negative_numeric_index(*element, idx)?);
    }
    let start_row = indices[0];
    let start_col = indices[1];
    let (end_row, end_col) = if indices.len() == 4 {
        (Some(indices[2]), Some(indices[3]))
    } else {
        (None, None)
    };
    let spec = RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    };
    validate_range(spec)
}

pub(super) fn non_negative_numeric_index(
    value: NumericScalar,
    position: usize,
) -> BuiltinResult<usize> {
    if let Some(integer) = value.into_int_value() {
        return integer.try_to_usize().ok_or_else(|| {
            dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                format!(
                    "dlmread: Range index {} is too large or negative",
                    position + 1
                ),
            )
        });
    }
    non_negative_index(value.materialize_f64(), position)
}

pub(super) fn non_negative_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range indices must be finite",
        ));
    }
    if value < 0.0 {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range indices must be non-negative",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range indices must be integers",
        ));
    }
    usize::try_from(rounded as i64).map_err(|_| {
        dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            format!(
                "dlmread: Range index {} is too large to fit in usize",
                position + 1
            ),
        )
    })
}

#[derive(Clone, Copy)]
pub(super) struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

pub(super) fn parse_cell_reference(token: &str) -> BuiltinResult<CellReference> {
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
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                format!("dlmread: invalid Range component '{token}'"),
            ));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range references cannot be empty",
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
            dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                format!(
                    "dlmread: invalid row index '{}' in Range component '{token}'",
                    digits
                ),
            )
        })?;
        if parsed == 0 {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                "dlmread: Range rows must be >= 1",
            ));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

pub(super) fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(dlmread_error_with(
                &DLMREAD_ERROR_RANGE,
                format!("dlmread: invalid column designator '{letters}' in Range"),
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| {
                dlmread_error_with(
                    &DLMREAD_ERROR_RANGE,
                    "dlmread: Range column index overflowed",
                )
            })?;
    }
    value.checked_sub(1).ok_or_else(|| {
        dlmread_error_with(
            &DLMREAD_ERROR_RANGE,
            "dlmread: Range column index underflowed",
        )
    })
}
