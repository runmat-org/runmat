use super::*;
use calamine::Data as SpreadsheetData;

pub(super) const MAX_EXCEL_ROW_INDEX: usize = 1_048_575;
const MAX_EXCEL_COLUMN_INDEX: usize = 16_383;
const MAX_XLSREAD_SELECTED_CELLS: usize = 5_000_000;

#[derive(Clone, Copy, Debug)]
pub(super) struct RangeSpec {
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) end_row: Option<usize>,
    pub(super) end_col: Option<usize>,
}

impl RangeSpec {
    fn validate(&self) -> BuiltinResult<()> {
        if self.start_row > MAX_EXCEL_ROW_INDEX {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range start row exceeds Excel limits",
            ));
        }
        if self.start_col > MAX_EXCEL_COLUMN_INDEX {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range start column exceeds Excel limits",
            ));
        }
        if let Some(end_row) = self.end_row {
            if end_row < self.start_row {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end row must be >= start row",
                ));
            }
            if end_row > MAX_EXCEL_ROW_INDEX {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end row exceeds Excel limits",
                ));
            }
        }
        if let Some(end_col) = self.end_col {
            if end_col < self.start_col {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end column must be >= start column",
                ));
            }
            if end_col > MAX_EXCEL_COLUMN_INDEX {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end column exceeds Excel limits",
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            parse_range_string(&value_to_string_scalar(value)?)
        }
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                parse_integer_range(&storage.exact_values())
            } else {
                parse_numeric_range(&t.materialize_f64())
            }
        }
        _ => Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range must be a string or numeric vector",
        )),
    }
}

pub(super) fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range string must not be empty",
        ));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: invalid range specification '{text}'"),
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range must specify a starting column",
        ));
    }
    let has_explicit_end = parts.len() == 2;
    let end = if has_explicit_end {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(end_ref) = end {
        if end_ref.col.is_none() {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range end must include a column reference",
            ));
        }
    }
    let mut end_row = end.and_then(|cell| cell.row);
    let mut end_col = end.and_then(|cell| cell.col);
    if !has_explicit_end && start.row.is_some() && start.col.is_some() {
        end_row = start.row;
        end_col = start.col;
    }
    let spec = RangeSpec {
        start_row: start.row.unwrap_or(0),
        start_col: start.col.unwrap(),
        end_row,
        end_col,
    };
    spec.validate()?;
    Ok(spec)
}

fn parse_numeric_range(values: &[f64]) -> BuiltinResult<RangeSpec> {
    if values.len() != 2 && values.len() != 4 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: numeric range must contain 2 or 4 one-based indices",
        ));
    }
    let indices = values
        .iter()
        .enumerate()
        .map(|(idx, value)| positive_index(*value, idx))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let spec = RangeSpec {
        start_row: indices[0],
        start_col: indices[1],
        end_row: if indices.len() == 4 {
            Some(indices[2])
        } else {
            None
        },
        end_col: if indices.len() == 4 {
            Some(indices[3])
        } else {
            None
        },
    };
    spec.validate()?;
    Ok(spec)
}

fn parse_integer_range(values: &[IntValue]) -> BuiltinResult<RangeSpec> {
    if values.len() != 2 && values.len() != 4 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: numeric range must contain 2 or 4 one-based indices",
        ));
    }
    let indices = values
        .iter()
        .enumerate()
        .map(|(idx, value)| positive_integer_index(value, idx))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let spec = RangeSpec {
        start_row: indices[0],
        start_col: indices[1],
        end_row: if indices.len() == 4 {
            Some(indices[2])
        } else {
            None
        },
        end_col: if indices.len() == 4 {
            Some(indices[3])
        } else {
            None
        },
    };
    spec.validate()?;
    Ok(spec)
}

fn positive_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range indices must be positive integers",
        ));
    }
    let rounded = value.round();
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: range index {} is too large", position + 1),
        ));
    }
    let one_based = rounded as usize;
    one_based.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range indices must be one-based",
        )
    })
}

fn positive_integer_index(value: &IntValue, position: usize) -> BuiltinResult<usize> {
    let one_based = value.try_to_usize().ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range indices must be positive integers",
        )
    })?;
    one_based.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: range index {} must be one-based", position + 1),
        )
    })
}

#[derive(Clone, Copy)]
struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

fn parse_cell_reference(token: &str) -> BuiltinResult<CellReference> {
    let trimmed = token.trim();
    let stripped = trimmed.chars().filter(|&ch| ch != '$').collect::<String>();
    if let Some(reference) = parse_r1c1_reference(&stripped)? {
        return Ok(reference);
    }
    let mut letters = String::new();
    let mut digits = String::new();
    let mut saw_digit = false;
    for ch in trimmed.chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            if saw_digit {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    format!("xlsread: invalid range component '{token}'"),
                ));
            }
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            saw_digit = true;
            digits.push(ch);
        } else {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid range component '{token}'"),
            ));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range references cannot be empty",
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
            xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid row index '{digits}'"),
            )
        })?;
        if parsed == 0 {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range rows must be one-based",
            ));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn parse_r1c1_reference(token: &str) -> BuiltinResult<Option<CellReference>> {
    let upper = token.to_ascii_uppercase();
    let Some(rest) = upper.strip_prefix('R') else {
        return Ok(None);
    };
    let Some(split) = rest.find('C') else {
        return Ok(None);
    };
    let (row_text, col_with_marker) = rest.split_at(split);
    let col_text = &col_with_marker[1..];
    if rest[split + 1..].contains('C') {
        return Ok(None);
    }
    if row_text.is_empty()
        || col_text.is_empty()
        || !row_text.chars().all(|ch| ch.is_ascii_digit())
        || !col_text.chars().all(|ch| ch.is_ascii_digit())
    {
        return Ok(None);
    }
    let row = parse_one_based_r1c1_index(row_text, "row", token)?;
    let col = parse_one_based_r1c1_index(col_text, "column", token)?;
    Ok(Some(CellReference {
        row: Some(row),
        col: Some(col),
    }))
}

fn parse_one_based_r1c1_index(text: &str, label: &str, token: &str) -> BuiltinResult<usize> {
    let parsed = text.parse::<usize>().map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: invalid {label} index in '{token}'"),
        )
    })?;
    parsed.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: {label} index in '{token}' must be one-based"),
        )
    })
}

fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid column designator '{letters}'"),
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|current| current.checked_add(digit))
            .ok_or_else(|| {
                xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: range column overflow")
            })?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: range column underflow"))
        .and_then(|zero_based| {
            if zero_based <= MAX_EXCEL_COLUMN_INDEX {
                Ok(zero_based)
            } else {
                Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    format!("xlsread: range column '{letters}' exceeds Excel limits"),
                ))
            }
        })
}

pub(super) fn text_looks_like_range(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.contains(':')
        || trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
            && trimmed.chars().any(|ch| ch.is_ascii_digit())
}

pub(super) fn selected_cells(
    range: &calamine::Range<SpreadsheetData>,
    requested: Option<RangeSpec>,
) -> BuiltinResult<Vec<Vec<SpreadsheetData>>> {
    if range.is_empty() && requested.is_none() {
        return Ok(Vec::new());
    }
    let start = range.start().unwrap_or((0, 0));
    let end = range.end().unwrap_or((0, 0));
    let start_row = requested
        .map(|spec| checked_u32(spec.start_row, "range row"))
        .transpose()?
        .unwrap_or(start.0);
    let start_col = requested
        .map(|spec| checked_u32(spec.start_col, "range column"))
        .transpose()?
        .unwrap_or(start.1);
    let end_row = requested
        .and_then(|spec| spec.end_row)
        .map(|row| checked_u32(row, "range row"))
        .transpose()?
        .unwrap_or(end.0);
    let end_col = requested
        .and_then(|spec| spec.end_col)
        .map(|col| checked_u32(col, "range column"))
        .transpose()?
        .unwrap_or(end.1);
    if start_row > end_row || start_col > end_col {
        return Ok(Vec::new());
    }
    validate_selected_shape(start_row, start_col, end_row, end_col)?;

    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        let mut row = Vec::new();
        for col_idx in start_col..=end_col {
            row.push(
                range
                    .get_value((row_idx, col_idx))
                    .cloned()
                    .unwrap_or(SpreadsheetData::Empty),
            );
        }
        rows.push(row);
    }
    if requested.is_some() {
        Ok(rows)
    } else {
        trim_empty_edges(rows)
    }
}

fn checked_u32(value: usize, context: &str) -> BuiltinResult<u32> {
    u32::try_from(value).map_err(|_| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, format!("xlsread: {context} overflow"))
    })
}

fn validate_selected_shape(
    start_row: u32,
    start_col: u32,
    end_row: u32,
    end_col: u32,
) -> BuiltinResult<()> {
    let row_count = usize::try_from(end_row - start_row + 1).map_err(|_| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: selected row count overflow")
    })?;
    let col_count = usize::try_from(end_col - start_col + 1).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: selected column count overflow",
        )
    })?;
    let cell_count = row_count.checked_mul(col_count).ok_or_else(|| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: selected range is too large")
    })?;
    if cell_count > MAX_XLSREAD_SELECTED_CELLS {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!(
                "xlsread: selected range has {cell_count} cells, exceeding the RunMat limit of {MAX_XLSREAD_SELECTED_CELLS}"
            ),
        ));
    }
    Ok(())
}

fn trim_empty_edges(rows: Vec<Vec<SpreadsheetData>>) -> BuiltinResult<Vec<Vec<SpreadsheetData>>> {
    if rows.is_empty() {
        return Ok(rows);
    }
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut first_row = None;
    let mut last_row = None;
    let mut first_col = None;
    let mut last_col = None;

    for row in 0..row_count {
        for col in 0..col_count {
            let cell = rows
                .get(row)
                .and_then(|values| values.get(col))
                .unwrap_or(&SpreadsheetData::Empty);
            if !matches!(cell, SpreadsheetData::Empty) {
                first_row = Some(first_row.map_or(row, |current: usize| current.min(row)));
                last_row = Some(last_row.map_or(row, |current: usize| current.max(row)));
                first_col = Some(first_col.map_or(col, |current: usize| current.min(col)));
                last_col = Some(last_col.map_or(col, |current: usize| current.max(col)));
            }
        }
    }

    let (Some(first_row), Some(last_row), Some(first_col), Some(last_col)) =
        (first_row, last_row, first_col, last_col)
    else {
        return Ok(Vec::new());
    };
    let mut trimmed = Vec::new();
    for row in rows.iter().take(last_row + 1).skip(first_row) {
        let mut out = Vec::new();
        for col in first_col..=last_col {
            out.push(row.get(col).cloned().unwrap_or(SpreadsheetData::Empty));
        }
        trimmed.push(out);
    }
    Ok(trimmed)
}
