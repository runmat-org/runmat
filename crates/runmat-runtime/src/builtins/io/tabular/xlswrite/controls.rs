use super::*;

#[derive(Debug, Clone)]
pub(super) struct XlsWriteRequest {
    pub(super) sheet: SheetSelector,
    pub(super) range: RangeStart,
}

impl Default for XlsWriteRequest {
    fn default() -> Self {
        Self {
            sheet: SheetSelector::Default,
            range: RangeStart::default(),
        }
    }
}

impl XlsWriteRequest {
    pub(super) fn preflight_host_arguments(
        args: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<()> {
        if args.len() > 3 {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_INVALID_ARGUMENT,
                "xlswrite: expected filename, data, optional sheet, optional range, and optional '-basic'",
            ));
        }
        let host = |value: &Value| !matches!(value, Value::GpuTensor(_));
        match args.len() {
            1 if host(&args[0]) => {
                let single = &args[0];
                parse_single_selector(single, &mut Self::default())?;
            }
            2 => {
                let (sheet, range) = (&args[0], &args[1]);
                if host(sheet) {
                    parse_sheet_selector(sheet)?;
                }
                if host(range) {
                    parse_range_start(range)?;
                }
            }
            3 => {
                let (sheet, range, mode) = (&args[0], &args[1], &args[2]);
                if host(sheet) {
                    parse_sheet_selector(sheet)?;
                }
                if host(range) {
                    parse_range_start(range)?;
                }
                if host(mode) {
                    parse_basic_mode(mode)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        let mut request = Self::default();
        match args.len() {
            0 => {}
            1 => parse_single_selector(&args[0], &mut request)?,
            2 => {
                let (sheet, range) = (&args[0], &args[1]);
                request.sheet = parse_sheet_selector(sheet)?;
                request.range = parse_range_start(range)?;
            }
            3 => {
                let (sheet, range, mode) = (&args[0], &args[1], &args[2]);
                request.sheet = parse_sheet_selector(sheet)?;
                request.range = parse_range_start(range)?;
                parse_basic_mode(mode)?;
            }
            _ => {
                return Err(xlswrite_error_with(
                    &XLSWRITE_ERROR_INVALID_ARGUMENT,
                    "xlswrite: expected filename, data, optional sheet, optional range, and optional '-basic'",
                ))
            }
        }
        Ok(request)
    }
}

#[derive(Debug, Clone)]
pub(super) enum SheetSelector {
    Default,
    Name(String),
    Index(usize),
}

fn parse_single_selector(value: &Value, request: &mut XlsWriteRequest) -> BuiltinResult<()> {
    if let Ok(text) = value_to_string_scalar(value) {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("basic") || trimmed.eq_ignore_ascii_case("-basic") {
            return Ok(());
        }
        if text_looks_like_range(trimmed) {
            request.range = parse_range_text(trimmed)?;
        } else {
            request.sheet = SheetSelector::Name(nonempty_sheet_name(trimmed)?);
        }
        return Ok(());
    }
    request.sheet = parse_sheet_selector(value)?;
    Ok(())
}

pub(super) fn parse_sheet_selector(value: &Value) -> BuiltinResult<SheetSelector> {
    match value {
        Value::Num(n) => numeric_sheet_index(*n),
        Value::Int(i) => {
            let index = integer_sheet_index(i)?;
            Ok(SheetSelector::Index(index))
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                let index = integer_sheet_index(&value)?;
                Ok(SheetSelector::Index(index))
            } else {
                numeric_sheet_index(tensor::tensor_value_f64(t, 0))
            }
        }
        _ => {
            let text = value_to_string_scalar(value).map_err(|_| {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_SHEET,
                    "xlswrite: sheet must be a name or one-based numeric index",
                )
            })?;
            Ok(SheetSelector::Name(nonempty_sheet_name(text.trim())?))
        }
    }
}

fn integer_sheet_index(value: &IntValue) -> BuiltinResult<usize> {
    let index = value
        .try_to_usize()
        .and_then(|index| index.checked_sub(1))
        .ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_SHEET,
                "xlswrite: sheet index must be one-based",
            )
        })?;
    if index >= MAX_XLSWRITE_SHEETS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            format!("xlswrite: sheet index exceeds supported limit of {MAX_XLSWRITE_SHEETS}"),
        ));
    }
    Ok(index)
}

fn numeric_sheet_index(value: f64) -> BuiltinResult<SheetSelector> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet index must be a positive integer",
        ));
    }
    let rounded = value.round();
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet index is too large",
        ));
    }
    let index = rounded as usize;
    let zero_based = index.checked_sub(1).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet index must be one-based",
        )
    })?;
    let index = zero_based;
    if index >= MAX_XLSWRITE_SHEETS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            format!("xlswrite: sheet index exceeds supported limit of {MAX_XLSWRITE_SHEETS}"),
        ));
    }
    Ok(SheetSelector::Index(index))
}

fn nonempty_sheet_name(text: &str) -> BuiltinResult<String> {
    validate_sheet_name(text)
}

pub(super) fn validate_sheet_name(text: &str) -> BuiltinResult<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name must not be empty",
        ));
    }
    if trimmed.chars().count() > 31 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name must be 31 characters or fewer",
        ));
    }
    if trimmed
        .chars()
        .any(|ch| matches!(ch, ':' | '\\' | '/' | '?' | '*' | '[' | ']'))
    {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name contains characters Excel does not allow",
        ));
    }
    Ok(trimmed.to_string())
}

pub(super) fn parse_range_start(value: &Value) -> BuiltinResult<RangeStart> {
    if let Ok(text) = value_to_string_scalar(value) {
        return parse_range_text(&text);
    }
    match value {
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                parse_integer_range_start(&storage.exact_values())
            } else {
                parse_numeric_range(&tensor::tensor_values_f64(t))
            }
        }
        Value::Int(value) => parse_integer_range_start(std::slice::from_ref(value)),
        Value::Num(n) => parse_numeric_range(&[*n]),
        _ => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range must be an A1 string or numeric [row col] vector",
        )),
    }
}

fn parse_integer_range_start(values: &[IntValue]) -> BuiltinResult<RangeStart> {
    if values.len() < 2 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: numeric range must contain at least row and column",
        ));
    }
    let row = integer_one_based_index(&values[0], "row", MAX_EXCEL_ROW_INDEX + 1)?;
    let col = integer_one_based_index(&values[1], "column", MAX_EXCEL_COLUMN_INDEX + 1)?;
    Ok(RangeStart { row, col })
}

fn integer_one_based_index(value: &IntValue, label: &str, maximum: usize) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value >= 1 && *value <= maximum)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                format!("xlswrite: range {label} must be within worksheet limits"),
            )
        })
}

fn parse_numeric_range(values: &[f64]) -> BuiltinResult<RangeStart> {
    if values.len() < 2 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: numeric range must contain at least row and column",
        ));
    }
    let row = one_based_index(values[0], "row")?;
    let col = one_based_index(values[1], "column")?;
    Ok(RangeStart { row, col })
}

fn one_based_index(value: f64, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: range {label} must be a positive integer"),
        ));
    }
    let rounded = value.round();
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: range {label} is too large"),
        ));
    }
    let one_based = rounded as usize;
    one_based.checked_sub(1).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: range {label} is too large"),
        )
    })
}

fn parse_range_text(value: &str) -> BuiltinResult<RangeStart> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range string must not be empty",
        ));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: invalid range specification '{value}'"),
        ));
    }
    let start = parse_a1_cell(parts[0].trim()).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range must start with an Excel A1 cell reference",
        )
    })?;
    if start.row > MAX_EXCEL_ROW_INDEX || start.col > MAX_EXCEL_COLUMN_INDEX {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range start exceeds Excel worksheet limits",
        ));
    }
    if parts.len() == 2 {
        let end = parse_a1_cell(parts[1].trim()).ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end must be an Excel A1 cell reference",
            )
        })?;
        if end.row < start.row || end.col < start.col {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end must be greater than or equal to range start",
            ));
        }
        if end.row > MAX_EXCEL_ROW_INDEX || end.col > MAX_EXCEL_COLUMN_INDEX {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end exceeds Excel worksheet limits",
            ));
        }
    }
    Ok(start)
}

fn parse_basic_mode(value: &Value) -> BuiltinResult<()> {
    let text = value_to_string_scalar(value)?;
    let normalized = text.trim().to_ascii_lowercase();
    if normalized == "basic" || normalized == "-basic" {
        return Ok(());
    }
    Err(xlswrite_error_with(
        &XLSWRITE_ERROR_INVALID_ARGUMENT,
        "xlswrite: only the legacy '-basic' mode flag is supported",
    ))
}

fn text_looks_like_range(value: &str) -> bool {
    parse_a1_cell(value.split(':').next().unwrap_or("")).is_some()
}

fn parse_a1_cell(value: &str) -> Option<RangeStart> {
    if value.is_empty() {
        return None;
    }
    let mut col = 0usize;
    let mut letters = 0usize;
    let stripped = value.chars().filter(|&ch| ch != '$').collect::<String>();
    for ch in stripped.chars() {
        if ch.is_ascii_alphabetic() {
            if letters == 0 && col != 0 {
                return None;
            }
            col = col.checked_mul(26)?;
            col = col.checked_add((ch.to_ascii_uppercase() as u8 - b'A' + 1) as usize)?;
            letters += 1;
        } else {
            break;
        }
    }
    let row_text = &stripped[letters..];
    if letters == 0 || row_text.is_empty() || !row_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let row: usize = row_text.parse().ok()?;
    if row == 0 || col == 0 {
        return None;
    }
    Some(RangeStart {
        row: row - 1,
        col: col - 1,
    })
}
