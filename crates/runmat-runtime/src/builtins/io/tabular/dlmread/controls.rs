use super::*;

#[derive(Clone, Debug)]
pub(super) enum DelimiterSpec {
    Auto,
    Char(char),
    String(String),
}

impl DelimiterSpec {
    fn new_from_string(raw: &str) -> BuiltinResult<Self> {
        if raw.is_empty() {
            return Ok(DelimiterSpec::Auto);
        }
        if raw == r"\t" {
            return Ok(DelimiterSpec::Char('\t'));
        }
        if raw == r"\n" {
            return Ok(DelimiterSpec::Char('\n'));
        }
        if raw == r"\r" {
            return Ok(DelimiterSpec::Char('\r'));
        }
        let mut chars = raw.chars();
        if let Some(first) = chars.next() {
            if chars.next().is_none() {
                return Ok(DelimiterSpec::Char(first));
            }
        }
        Ok(DelimiterSpec::String(raw.to_string()))
    }

    pub(super) fn split<'a>(&'a self, line: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        match self {
            DelimiterSpec::Auto if line.contains(',') => Box::new(line.split(',')),
            DelimiterSpec::Auto => Box::new(line.split_whitespace()),
            DelimiterSpec::Char(ch) => Box::new(line.split(*ch)),
            DelimiterSpec::String(pattern) => Box::new(line.split(pattern.as_str())),
        }
    }
}

#[derive(Debug)]
pub(super) struct DlmReadOptions {
    pub(super) delimiter: DelimiterSpec,
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) range: Option<RangeSpec>,
}

impl Default for DlmReadOptions {
    fn default() -> Self {
        Self {
            delimiter: DelimiterSpec::Auto,
            start_row: 0,
            start_col: 0,
            range: None,
        }
    }
}

pub(super) async fn parse_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<DlmReadOptions> {
    let gathered = args
        .try_map_preserving_roots_async(args.len(), runtime, async |index| {
            let value = await_io(runtime, BUILTIN_NAME, gather_if_needed_async(&args[index]))
                .await?
                .map_err(map_control_flow)?;
            cooperation.completed(0).await?;
            Ok(value)
        })
        .await?;

    let mut options = DlmReadOptions::default();

    match gathered.len() {
        0 => Ok(options),
        1 => {
            if is_range_candidate(&gathered[0]) {
                require_composed_range_extension()?;
                options.range = Some(parse_range(&gathered[0])?);
            } else {
                options.delimiter = parse_delimiter(&gathered[0])?;
            }
            Ok(options)
        }
        2 => {
            if is_range_candidate(&gathered[1]) {
                options.delimiter = parse_delimiter(&gathered[0])?;
                options.range = Some(parse_range(&gathered[1])?);
            } else {
                require_composed_range_extension()?;
                options.start_row = value_to_start_index(&gathered[0], "row")?;
                options.start_col = value_to_start_index(&gathered[1], "col")?;
            }
            Ok(options)
        }
        3 => {
            if is_range_candidate(&gathered[2]) {
                require_composed_range_extension()?;
                options.start_row = value_to_start_index(&gathered[0], "row")?;
                options.start_col = value_to_start_index(&gathered[1], "col")?;
                options.range = Some(parse_range(&gathered[2])?);
            } else if is_delimiter_value(&gathered[0]) {
                options.delimiter = parse_delimiter(&gathered[0])?;
                options.start_row = value_to_start_index(&gathered[1], "row")?;
                options.start_col = value_to_start_index(&gathered[2], "col")?;
            } else {
                return Err(dlmread_error_with(
                    &DLMREAD_ERROR_ARG_CONFIG,
                    "dlmread: expected dlmread(filename[, delimiter][, row, col][, range])",
                ));
            }
            Ok(options)
        }
        4 => {
            require_composed_range_extension()?;
            if !is_range_candidate(&gathered[3]) {
                return Err(dlmread_error_with(
                    &DLMREAD_ERROR_ARG_CONFIG,
                    "dlmread: expected Range as final argument",
                ));
            }
            options.delimiter = parse_delimiter(&gathered[0])?;
            options.start_row = value_to_start_index(&gathered[1], "row")?;
            options.start_col = value_to_start_index(&gathered[2], "col")?;
            options.range = Some(parse_range(&gathered[3])?);
            Ok(options)
        }
        _ => Err(dlmread_error_with(
            &DLMREAD_ERROR_ARG_CONFIG,
            "dlmread: expected dlmread(filename[, delimiter][, row, col][, range])",
        )),
    }
}

pub(super) fn require_composed_range_extension() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &DLMREAD_COMPOSED_RANGE_EXTENSION,
        BUILTIN_NAME,
    )
}

pub(super) fn is_delimiter_value(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => true,
        Value::Int(_) | Value::Num(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        _ => false,
    }
}

pub(super) fn is_range_candidate(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            crate::builtins::common::validation::value_text_scalar_string(value)
                .is_some_and(|text| looks_like_range_string(&text))
        }
        Value::Tensor(t) => tensor_len(t) == 2 || tensor_len(t) == 4,
        _ => false,
    }
}

pub(super) fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

pub(super) fn looks_like_range_string(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed.chars().any(|ch| ch.is_ascii_digit()) || trimmed.contains(':')
}

pub(super) fn parse_delimiter(value: &Value) -> BuiltinResult<DelimiterSpec> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return DelimiterSpec::new_from_string(&text);
    }
    match value {
        Value::StringArray(_) => Err(dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            "dlmread: string array delimiters must be scalar",
        )),
        Value::Int(i) => {
            require_numeric_delimiter_extension()?;
            delimiter_from_integer(i)
        }
        Value::Num(n) => {
            require_numeric_delimiter_extension()?;
            delimiter_from_numeric(*n)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            require_numeric_delimiter_extension()?;
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                delimiter_from_integer(&value)
            } else {
                delimiter_from_numeric(tensor::tensor_value_f64(t, 0))
            }
        }
        _ => Err(dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            format!("dlmread: unsupported delimiter value {value:?}"),
        )),
    }
}

pub(super) fn require_numeric_delimiter_extension() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &DLMREAD_NUMERIC_DELIMITER_EXTENSION,
        BUILTIN_NAME,
    )
}

pub(super) fn delimiter_from_integer(value: &IntValue) -> BuiltinResult<DelimiterSpec> {
    value.try_to_i64().map_or_else(
        || {
            Err(dlmread_error_with(
                &DLMREAD_ERROR_DELIMITER,
                "dlmread: delimiter code must be within Unicode range",
            ))
        },
        delimiter_from_ascii,
    )
}

pub(super) fn delimiter_from_numeric(value: f64) -> BuiltinResult<DelimiterSpec> {
    if !value.is_finite() {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            "dlmread: delimiter code must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            "dlmread: delimiter code must be an integer",
        ));
    }
    delimiter_from_ascii(rounded as i64)
}

pub(super) fn delimiter_from_ascii(value: i64) -> BuiltinResult<DelimiterSpec> {
    if value < 0 || value > char::MAX as i64 {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            "dlmread: delimiter code must be within Unicode range",
        ));
    }
    let ch = char::from_u32(value as u32).ok_or_else(|| {
        dlmread_error_with(
            &DLMREAD_ERROR_DELIMITER,
            "dlmread: delimiter code does not map to a Unicode scalar",
        )
    })?;
    Ok(DelimiterSpec::Char(ch))
}

pub(super) fn value_to_start_index(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => i.try_to_usize().ok_or_else(|| {
            dlmread_error_with(
                &DLMREAD_ERROR_INDEX,
                format!("dlmread: {name} must be a non-negative integer"),
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(dlmread_error_with(
                    &DLMREAD_ERROR_INDEX,
                    format!("dlmread: {name} must be finite"),
                ));
            }
            if *n < 0.0 {
                return Err(dlmread_error_with(
                    &DLMREAD_ERROR_INDEX,
                    format!("dlmread: {name} must be a non-negative integer"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(dlmread_error_with(
                    &DLMREAD_ERROR_INDEX,
                    format!("dlmread: {name} must be an integer"),
                ));
            }
            usize::try_from(rounded as i64).map_err(|_| {
                dlmread_error_with(
                    &DLMREAD_ERROR_INDEX,
                    format!("dlmread: {name} is too large"),
                )
            })
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                value.try_to_usize().ok_or_else(|| {
                    dlmread_error_with(
                        &DLMREAD_ERROR_INDEX,
                        format!("dlmread: {name} must be a non-negative integer"),
                    )
                })
            } else {
                value_to_start_index(&Value::Num(tensor::tensor_value_f64(t, 0)), name)
            }
        }
        _ => Err(dlmread_error_with(
            &DLMREAD_ERROR_INDEX,
            format!("dlmread: expected numeric scalar for {name}, got {value:?}"),
        )),
    }
}

pub(super) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return normalize_path(&text);
    }
    match value {
        Value::StringArray(_) => Err(dlmread_error_with(
            &DLMREAD_ERROR_FILENAME,
            "dlmread: string array filename inputs must be scalar",
        )),
        Value::CharArray(_) => Err(dlmread_error_with(
            &DLMREAD_ERROR_FILENAME,
            "dlmread: expected a 1-by-N character vector for the file name",
        )),
        other => Err(dlmread_error_with(
            &DLMREAD_ERROR_FILENAME,
            format!(
                "dlmread: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

pub(super) fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(dlmread_error(&DLMREAD_ERROR_FILENAME_EMPTY));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|message| dlmread_error_with(&DLMREAD_ERROR_FILENAME, message))?;
    Ok(Path::new(&expanded).to_path_buf())
}
