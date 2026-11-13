use super::*;

pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<WriteMatrixOptions> {
    if args.is_empty() {
        return Ok(WriteMatrixOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG));
    }

    let mut options = WriteMatrixOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name = option_name_from_value(&args[index])?;
        apply_option(&mut options, &name, &args[index + 1])?;
        index += 2;
    }
    Ok(options)
}

pub(super) fn preflight_host_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut checked = WriteMatrixOptions::default();
    for pair in args.chunks_exact(2) {
        if matches!(pair[0], Value::GpuTensor(_)) {
            return Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_OPTION,
                "writematrix: option name must be scalar text",
            ));
        }
        let name = option_name_from_value(&pair[0])?;
        if !matches!(pair[1], Value::GpuTensor(_)) {
            apply_option(&mut checked, &name, &pair[1])?;
        }
    }
    Ok(())
}

fn apply_option(options: &mut WriteMatrixOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    if name.eq_ignore_ascii_case("Delimiter") {
        let delimiter = parse_delimiter(value)?;
        options.delimiter = Some(delimiter);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("WriteMode") {
        options.write_mode = parse_write_mode(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("QuoteStrings") {
        options.quote_strings = parse_quote_strings(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("DecimalSeparator") {
        options.decimal_separator = parse_decimal_separator(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("LineEnding") {
        options.line_ending = parse_line_ending(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("FileType") {
        options.file_type = parse_file_type(value)?;
        return Ok(());
    }
    // Unsupported or future options are ignored for compatibility with MATLAB's permissive behaviour.
    Ok(())
}

pub(super) fn option_name_from_value(value: &Value) -> BuiltinResult<String> {
    value_to_string_scalar(value, "option name")
}

fn parse_delimiter(value: &Value) -> BuiltinResult<String> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: Delimiter cannot be empty",
        ));
    }
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "tab" => Ok("\t".to_string()),
        "space" | "whitespace" => Ok(" ".to_string()),
        "comma" => Ok(",".to_string()),
        "semicolon" => Ok(";".to_string()),
        "pipe" => Ok("|".to_string()),
        _ => Ok(trimmed.to_string()),
    }
}

fn parse_write_mode(value: &Value) -> BuiltinResult<WriteMode> {
    let text = value_to_string_scalar(value, "WriteMode")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: WriteMode must be 'overwrite' or 'append'",
        )),
    }
}

fn parse_quote_strings(value: &Value) -> BuiltinResult<QuoteStrings> {
    match value {
        Value::Bool(value) => Ok(if *value {
            QuoteStrings::All
        } else {
            QuoteStrings::None
        }),
        Value::Int(i) => {
            let raw = i.to_i64();
            match raw {
                0 => Ok(QuoteStrings::None),
                1 => Ok(QuoteStrings::All),
                _ => Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    "writematrix: QuoteStrings numeric aliases must be 0 or 1",
                )),
            }
        }
        Value::Num(n) => {
            if (*n - 0.0).abs() < f64::EPSILON {
                Ok(QuoteStrings::None)
            } else if (*n - 1.0).abs() < f64::EPSILON {
                Ok(QuoteStrings::All)
            } else {
                Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    "writematrix: QuoteStrings numeric aliases must be 0 or 1",
                ))
            }
        }
        _ => {
            let text = value_to_string_scalar(value, "QuoteStrings")?;
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "minimal" => Ok(QuoteStrings::Minimal),
                "all" | "on" | "true" | "yes" | "1" => Ok(QuoteStrings::All),
                "none" | "off" | "false" | "no" | "0" => Ok(QuoteStrings::None),
                _ => Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    "writematrix: QuoteStrings must be 'minimal', 'all', or 'none'",
                )),
            }
        }
    }
}

fn parse_decimal_separator(value: &Value) -> BuiltinResult<char> {
    let text = value_to_string_scalar(value, "DecimalSeparator")?;
    let mut chars = text.chars();
    let ch = chars.next().ok_or_else(|| {
        writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator must be a single character",
        )
    })?;
    if chars.next().is_some() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator must be a single character",
        ));
    }
    if ch == '\n' || ch == '\r' {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator cannot be a newline character",
        ));
    }
    Ok(ch)
}

fn parse_line_ending(value: &Value) -> BuiltinResult<LineEnding> {
    let text = value_to_string_scalar(value, "LineEnding")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: LineEnding must be 'auto', 'unix', 'pc', or 'mac'",
        )),
    }
}

fn parse_file_type(value: &Value) -> BuiltinResult<FileType> {
    let text = value_to_string_scalar(value, "FileType")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "delimitedtext" | "text" => {
            if lowered == "text" {
                Ok(FileType::Text)
            } else {
                Ok(FileType::DelimitedText)
            }
        }
        "spreadsheet" => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: FileType 'spreadsheet' is not supported; export delimited text instead",
        )),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: unsupported FileType",
        )),
    }
}

pub(super) fn value_to_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            format!("writematrix: expected {context} as a string scalar or character vector"),
        )
    })
}
