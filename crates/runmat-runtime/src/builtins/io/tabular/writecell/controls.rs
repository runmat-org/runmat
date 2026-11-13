use super::*;

/// Reject all malformed host-known pairs before any resident option is read.
/// A resident name or value is deliberately left for the ordinary parser.
pub(super) fn preflight_host_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for pair in args.chunks_exact(2) {
        if matches!(&pair[0], Value::GpuTensor(_)) {
            continue;
        }
        let name = string_scalar_from_value(&pair[0], "option name")
            .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
        if matches!(&pair[1], Value::GpuTensor(_)) {
            continue;
        }
        apply_option(&mut WriteCellOptions::default(), &name, &pair[1])?;
    }
    Ok(())
}

pub(super) async fn parse_options(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<WriteCellOptions> {
    if args.is_empty() {
        return Ok(WriteCellOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG));
    }

    let mut options = WriteCellOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name_value = await_io(runtime, BUILTIN_NAME, gather_if_needed_async(&args[index]))
            .await?
            .map_err(map_control_flow)?;
        cooperation.completed(0).await?;
        let name = string_scalar_from_value(&name_value, "option name")
            .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
        let value = await_io(
            runtime,
            BUILTIN_NAME,
            gather_if_needed_async(&args[index + 1]),
        )
        .await?
        .map_err(map_control_flow)?;
        cooperation.completed(0).await?;
        apply_option(&mut options, &name, &value)?;
        index += 2;
    }
    Ok(options)
}

fn apply_option(options: &mut WriteCellOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    if name.eq_ignore_ascii_case("Delimiter") {
        options.delimiter = Some(parse_delimiter(value)?);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("WriteMode") {
        options.write_mode = parse_write_mode(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("QuoteStrings") {
        options.quote_strings = parse_bool_like(value, "QuoteStrings")?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("LineEnding") {
        options.line_ending = parse_line_ending(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("FileType") {
        options.file_type = Some(parse_file_type(value)?);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("Sheet") {
        options.sheet = parse_sheet(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("Range") {
        options.range = Some(parse_range_start(value)?);
        return Ok(());
    }
    Ok(())
}

fn parse_delimiter(value: &Value) -> BuiltinResult<String> {
    let text = string_scalar_from_value(value, "Delimiter")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    if text.is_empty() {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Delimiter cannot be empty",
        ));
    }
    let trimmed = text.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "tab" => Ok("\t".to_string()),
        "space" | "whitespace" => Ok(" ".to_string()),
        "comma" => Ok(",".to_string()),
        "semicolon" => Ok(";".to_string()),
        "pipe" => Ok("|".to_string()),
        _ => Ok(trimmed.to_string()),
    }
}

fn parse_write_mode(value: &Value) -> BuiltinResult<WriteMode> {
    let text = string_scalar_from_value(value, "WriteMode")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: WriteMode must be 'overwrite' or 'append'",
        )),
    }
}

fn parse_bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => match i.to_i64() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(writecell_error_with(
                &WRITECELL_ERROR_OPTION,
                format!("writecell: {context} must be logical (0 or 1)"),
            )),
        },
        Value::Num(n) if (*n - 0.0).abs() < f64::EPSILON => Ok(false),
        Value::Num(n) if (*n - 1.0).abs() < f64::EPSILON => Ok(true),
        _ => {
            let text = string_scalar_from_value(value, context)
                .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
            match text.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(writecell_error_with(
                    &WRITECELL_ERROR_OPTION,
                    format!("writecell: {context} must be logical (true/on or false/off)"),
                )),
            }
        }
    }
}

fn parse_line_ending(value: &Value) -> BuiltinResult<LineEnding> {
    let text = string_scalar_from_value(value, "LineEnding")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: LineEnding must be 'auto', 'unix', 'pc', or 'mac'",
        )),
    }
}

fn parse_file_type(value: &Value) -> BuiltinResult<OutputFileType> {
    let text = string_scalar_from_value(value, "FileType")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "text" | "delimitedtext" => Ok(OutputFileType::DelimitedText),
        "spreadsheet" => Ok(OutputFileType::Spreadsheet),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: FileType must be 'text', 'delimitedtext', or 'spreadsheet'",
        )),
    }
}

pub(super) fn parse_sheet(value: &Value) -> BuiltinResult<SheetSelector> {
    match value {
        Value::Num(n)
            if n.is_finite() && *n >= 1.0 && n.fract() == 0.0 && *n < usize::MAX as f64 =>
        {
            Ok(SheetSelector::Index(*n as usize))
        }
        Value::Int(i) => i
            .try_to_usize()
            .filter(|index| *index >= 1)
            .map(SheetSelector::Index)
            .ok_or_else(|| {
                writecell_error_with(
                    &WRITECELL_ERROR_OPTION,
                    "writecell: Sheet must be a name or one-based numeric index",
                )
            }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                value
                    .try_to_usize()
                    .filter(|index| *index >= 1)
                    .map(SheetSelector::Index)
                    .ok_or_else(|| {
                        writecell_error_with(
                            &WRITECELL_ERROR_OPTION,
                            "writecell: Sheet must be a name or one-based numeric index",
                        )
                    })
            } else {
                parse_sheet(&Value::Num(tensor::tensor_value_f64(tensor, 0)))
            }
        }
        _ => {
            let text = string_scalar_from_value(value, "Sheet")
                .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
            if text.trim().is_empty() {
                return Err(writecell_error_with(
                    &WRITECELL_ERROR_OPTION,
                    "writecell: Sheet name cannot be empty",
                ));
            }
            Ok(SheetSelector::Name(text))
        }
    }
}

fn parse_range_start(value: &Value) -> BuiltinResult<RangeStart> {
    let text = string_scalar_from_value(value, "Range")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    let start = text.split(':').next().unwrap_or("").trim();
    parse_a1_cell(start).ok_or_else(|| {
        writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Range must start with an Excel A1 cell reference",
        )
    })
}

fn parse_a1_cell(value: &str) -> Option<RangeStart> {
    if value.is_empty() {
        return None;
    }
    let mut col = 0usize;
    let mut letters = 0usize;
    for ch in value.chars() {
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
    let row_text = &value[letters..];
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
