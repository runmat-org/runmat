use super::*;

#[derive(Clone, Debug)]
pub(super) struct DlmWriteOptions {
    pub(super) delimiter: String,
    pub(super) newline: LineEnding,
    pub(super) roffset: usize,
    pub(super) coffset: usize,
    pub(super) precision: PrecisionSpec,
    pub(super) precision_explicit: bool,
    pub(super) append: bool,
}

impl Default for DlmWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: ",".to_string(),
            newline: LineEnding::platform_default(),
            roffset: 0,
            coffset: 0,
            precision: PrecisionSpec::Significant(5),
            precision_explicit: false,
            append: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum LineEnding {
    Unix,
    Pc,
    Mac,
}

impl LineEnding {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Unix => "\n",
            LineEnding::Pc => "\r\n",
            LineEnding::Mac => "\r",
        }
    }

    pub(super) fn platform_default() -> Self {
        if cfg!(windows) {
            LineEnding::Pc
        } else {
            LineEnding::Unix
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum PrecisionSpec {
    Significant(u32),
    Format(String),
}

pub(super) fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DlmWriteOptions> {
    let mut options = DlmWriteOptions::default();
    if args.is_empty() {
        return Ok(options);
    }

    let mut idx = 0usize;
    // Consume any leading '-append' flags.
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
        } else {
            break;
        }
    }

    // Optional positional delimiter.
    if idx < args.len() && is_positional_delimiter_candidate(&args[idx]) {
        options.delimiter = parse_delimiter_value(&args[idx])?;
        idx += 1;
        // Optional positional row/col offsets if both numeric scalars follow.
        if idx + 1 < args.len()
            && is_numeric_scalar(&args[idx])
            && is_numeric_scalar(&args[idx + 1])
        {
            options.roffset = parse_offset_value(&args[idx], "row offset")?;
            options.coffset = parse_offset_value(&args[idx + 1], "column offset")?;
            idx += 2;
        }
        // Consume any subsequent '-append'.
        while idx < args.len() {
            if is_append_flag(&args[idx]) {
                options.append = true;
                idx += 1;
            } else {
                break;
            }
        }
    }

    // Remaining arguments should be name-value pairs (allowing additional '-append').
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
            continue;
        }

        let name = value_to_lowercase_string(&args[idx])
            .ok_or_else(|| dlmwrite_error(&DLMWRITE_ERROR_ARG_CONFIG))?;
        idx += 1;
        if idx >= args.len() {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_ARG_CONFIG,
                "dlmwrite: name-value arguments must appear in pairs",
            ));
        }
        let value = &args[idx];
        idx += 1;

        match name.as_str() {
            "delimiter" => {
                options.delimiter = parse_delimiter_value(value)?;
            }
            "precision" => {
                options.precision = parse_precision_value(value)?;
                options.precision_explicit = true;
            }
            "newline" => {
                options.newline = parse_newline_value(value)?;
            }
            "roffset" => {
                options.roffset = parse_offset_value(value, "row offset")?;
            }
            "coffset" => {
                options.coffset = parse_offset_value(value, "column offset")?;
            }
            "append" => {
                options.append = parse_append_value(value)?;
            }
            other => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    format!("dlmwrite: unsupported name-value pair '{other}'"),
                ));
            }
        }
    }

    Ok(options)
}

pub(super) fn is_append_flag(value: &Value) -> bool {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("-append"))
}

pub(super) fn is_positional_delimiter_candidate(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            if let Some(lower) = value_to_lowercase_string(value) {
                match lower.as_str() {
                    "delimiter" | "precision" | "newline" | "roffset" | "coffset" | "append"
                    | "-append" => return false,
                    _ => {}
                }
            }
            true
        }
        _ => false,
    }
}

pub(super) fn parse_delimiter_value(value: &Value) -> BuiltinResult<String> {
    let text =
        crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_OPTION,
                "dlmwrite: delimiter must be a string scalar or character vector",
            )
        })?;
    interpret_delimiter_string(&text)
}

pub(super) fn interpret_delimiter_string(raw: &str) -> BuiltinResult<String> {
    if raw.is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: delimiter must not be empty",
        ));
    }
    if raw == r"\t" {
        return Ok("\t".to_string());
    }
    if raw == r"\n" {
        return Ok("\n".to_string());
    }
    if raw == r"\r" {
        return Ok("\r".to_string());
    }
    Ok(raw.to_string())
}

pub(super) fn is_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

pub(super) fn parse_offset_value(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => return integer_offset_value(value, context),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return integer_offset_value(&value, context);
            }
        }
        _ => {}
    }

    let scalar = extract_scalar(value).map_err(|e| {
        dlmwrite_error_with(&DLMWRITE_ERROR_OPTION, format!("dlmwrite: {context} {e}"))
    })?;
    if !scalar.is_finite() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be finite"),
        ));
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > 1e-9 {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be an integer, got {scalar}"),
        ));
    }
    if rounded < 0.0 {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be >= 0"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} is too large"),
        ));
    }
    Ok(rounded as usize)
}

pub(super) fn integer_offset_value(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be >= 0"),
        )
    })
}

pub(super) fn parse_precision_value(value: &Value) -> BuiltinResult<PrecisionSpec> {
    match value {
        Value::Int(i) => integer_precision_value(i),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision scalar must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1e-9 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision scalar must be an integer",
                ));
            }
            if rounded <= 0.0 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ));
            }
            if rounded > u32::MAX as f64 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ));
            }
            Ok(PrecisionSpec::Significant(rounded as u32))
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                integer_precision_value(&value)
            } else {
                parse_precision_value(&Value::Num(tensor::tensor_value_f64(t, 0)))
            }
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            if logical.data[0] != 0 {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ))
            }
        }
        Value::Bool(b) => {
            if *b {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ))
            }
        }
        _ => crate::builtins::common::validation::value_text_scalar_string(value)
            .ok_or_else(|| {
                dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be numeric or a format string",
                )
            })
            .and_then(|text| parse_precision_format(&text)),
    }
}

pub(super) fn integer_precision_value(value: &IntValue) -> BuiltinResult<PrecisionSpec> {
    let Some(digits) = value.try_to_u64() else {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: precision must be a positive integer",
        ));
    };
    if digits == 0 || digits > u32::MAX as u64 {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: precision must be a positive integer",
        ));
    }
    Ok(PrecisionSpec::Significant(digits as u32))
}

pub(super) fn parse_precision_format(text: &str) -> BuiltinResult<PrecisionSpec> {
    if text.is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: precision format string must not be empty",
        ));
    }
    Ok(PrecisionSpec::Format(text.to_string()))
}

pub(super) fn parse_newline_value(value: &Value) -> BuiltinResult<LineEnding> {
    let text = value_to_lowercase_string(value).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: newline must be a string scalar or character vector",
        )
    })?;
    match text.as_str() {
        "pc" | "windows" | "crlf" => Ok(LineEnding::Pc),
        "unix" | "lf" => Ok(LineEnding::Unix),
        "mac" | "cr" => Ok(LineEnding::Mac),
        other => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: unsupported newline setting '{other}' (expected 'pc' or 'unix')"),
        )),
    }
}

pub(super) fn parse_append_value(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        _ => crate::builtins::common::validation::value_text_scalar_string(value)
            .ok_or_else(|| {
                dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: append value must be logical",
                )
            })
            .and_then(|text| parse_bool_string(&text)),
    }
}

pub(super) fn parse_bool_string(text: &str) -> BuiltinResult<bool> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: append value must be logical",
        )),
    }
}

pub(super) fn extract_scalar(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "must be numeric scalar",
        )),
    }
}

pub(super) fn value_to_lowercase_string(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            crate::builtins::common::validation::value_text_scalar_string(value)
        }
        _ => tensor::value_to_string(value),
    }?;
    Some(text.trim().to_ascii_lowercase())
}

pub(super) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw =
        crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_FILENAME,
                "dlmwrite: filename must be a string scalar or character vector",
            )
        })?;
    if raw.trim().is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_FILENAME,
            "dlmwrite: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(&raw, BUILTIN_NAME)
        .map_err(|msg| dlmwrite_error_with(&DLMWRITE_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

pub(super) fn ensure_matrix_shape(tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape().len() <= 2 {
        return Ok(());
    }
    if tensor.shape()[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(dlmwrite_error_with(
        &DLMWRITE_ERROR_DATA_SHAPE,
        "dlmwrite: input must be 2-D; reshape before writing",
    ))
}
