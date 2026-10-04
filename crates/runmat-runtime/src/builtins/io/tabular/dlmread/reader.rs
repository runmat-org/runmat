use super::*;

pub(super) async fn read_dlm_rows(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    delimiter: &DelimiterSpec,
    parse_start_row: usize,
    parse_start_col: usize,
) -> BuiltinResult<(Vec<Vec<f64>>, usize)> {
    let path = selected.path();
    let file = await_io(runtime, BUILTIN_NAME, selected.open_read())
        .await?
        .map_err(|err| {
            dlmread_error_with_source(
                &DLMREAD_ERROR_IO_OPEN,
                format!("dlmread: unable to open '{}': {err}", path.display()),
                err,
            )
        })?;
    let mut reader = BufReader::new(file);
    let mut buffer = String::new();
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    let mut line_index = 0usize;

    loop {
        selected.check().map_err(|source| {
            dlmread_error_with_source(
                &DLMREAD_ERROR_IO_READ,
                "dlmread: captured filesystem authority was revoked",
                source,
            )
        })?;
        buffer.clear();
        let bytes = reader.read_line(&mut buffer).map_err(|err| {
            dlmread_error_with_source(
                &DLMREAD_ERROR_IO_READ,
                format!("dlmread: failed to read '{}': {err}", path.display()),
                err,
            )
        })?;
        if bytes == 0 {
            break;
        }
        cooperation.completed(bytes).await?;
        // Trim standard newline endings.
        if buffer.ends_with('\n') {
            buffer.pop();
            if buffer.ends_with('\r') {
                buffer.pop();
            }
        } else if buffer.ends_with('\r') {
            buffer.pop();
        }
        let mut view: &str = &buffer;
        if line_index == 0 && view.starts_with('\u{FEFF}') {
            view = &view['\u{FEFF}'.len_utf8()..];
        }
        if line_index < parse_start_row {
            rows.push(Vec::new());
            line_index += 1;
            continue;
        }
        let parsed =
            parse_dlm_row(cooperation, view, delimiter, line_index, parse_start_col).await?;
        max_cols = max_cols.max(parsed.len());
        rows.push(parsed);
        line_index += 1;
    }

    selected.check().map_err(|source| {
        dlmread_error_with_source(
            &DLMREAD_ERROR_IO_READ,
            "dlmread: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok((rows, max_cols))
}

pub(super) async fn parse_dlm_row(
    cooperation: &mut TabularCooperation<'_>,
    line: &str,
    delimiter: &DelimiterSpec,
    line_index: usize,
    parse_start_col: usize,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    for (col_index, raw_field) in delimiter.split(line).enumerate() {
        if col_index > 0 && col_index % 1024 == 0 {
            cooperation.completed(1024).await?;
        }
        if col_index < parse_start_col {
            // Skip validation for columns that will be dropped by row/column
            // offsets or range start-column semantics.
            values.push(0.0);
            continue;
        }
        let trimmed = raw_field.trim();
        if trimmed.is_empty() {
            values.push(0.0);
            continue;
        }
        let lowered = trimmed.to_ascii_lowercase();
        let value = match lowered.as_str() {
            "nan" => f64::NAN,
            "inf" | "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            _ => trimmed.parse::<f64>().map_err(|_| {
                dlmread_error_with(
                    &DLMREAD_ERROR_NON_NUMERIC_TOKEN,
                    format!(
                        "dlmread: nonnumeric token '{}' at row {}, column {}",
                        trimmed,
                        line_index + 1,
                        col_index + 1
                    ),
                )
            })?,
        };
        values.push(value);
    }
    cooperation.completed(values.len() % 1024).await?;
    Ok(values)
}
