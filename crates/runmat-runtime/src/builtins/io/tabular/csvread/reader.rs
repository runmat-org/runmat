use super::*;
use crate::builtins::io::tabular::line_reader::{read_line_bytes, LineReadError};

pub(super) async fn read_csv_rows(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    options: &CsvReadOptions,
) -> BuiltinResult<(Vec<Vec<ParsedNumber>>, usize, usize)> {
    let path = selected.path();
    let file = await_io(runtime, BUILTIN_NAME, selected.open_read())
        .await?
        .map_err(|err| {
            csvread_error_with_source(
                &CSVREAD_ERROR_IO_OPEN,
                format!("csvread: unable to open '{}': {err}", path.display()),
                err,
            )
        })?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    let mut line_index = 0usize;
    let mut nonempty_row_index = 0usize;
    let mut skipped_rows = 0usize;

    loop {
        selected.check().map_err(|source| {
            csvread_error_with_source(
                &CSVREAD_ERROR_IO_READ,
                "csvread: captured filesystem authority was revoked",
                source,
            )
        })?;
        let bytes = read_line_bytes(&mut reader, &mut buffer, cooperation)
            .await
            .map_err(|error| match error {
                LineReadError::Io(source) => csvread_error_with_source(
                    &CSVREAD_ERROR_IO_READ,
                    format!("csvread: failed to read '{}': {source}", path.display()),
                    source,
                ),
                LineReadError::Cooperation(source) => *source,
            })?;
        if bytes == 0 {
            break;
        }
        line_index += 1;
        trim_line_ending(&mut buffer);
        if buffer.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let current_row_index = nonempty_row_index;
        nonempty_row_index += 1;
        if let Some(range) = options.range {
            if let Some(end_row) = range.end_row {
                if current_row_index > end_row {
                    break;
                }
            }
            if current_row_index < range.start_row {
                continue;
            }
        }
        if options.range.is_none() && options.start_row > 0 && line_index <= options.start_row {
            skipped_rows += 1;
            continue;
        }
        let (parse_start_col, parse_end_col) = match options.range {
            Some(range) => (range.start_col, range.end_col),
            None => (options.start_col, None),
        };
        let line = String::from_utf8_lossy(&buffer);
        let parsed = parse_csv_row(
            cooperation,
            &line,
            line_index,
            parse_start_col,
            parse_end_col,
        )
        .await?;
        max_cols = max_cols.max(parsed.len());
        rows.push(parsed);
    }

    selected.check().map_err(|source| {
        csvread_error_with_source(
            &CSVREAD_ERROR_IO_READ,
            "csvread: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok((rows, max_cols, skipped_rows))
}

fn trim_line_ending(buffer: &mut Vec<u8>) {
    if buffer.ends_with(b"\n") {
        buffer.pop();
        if buffer.ends_with(b"\r") {
            buffer.pop();
        }
    } else if buffer.ends_with(b"\r") {
        buffer.pop();
    }
}

async fn parse_csv_row(
    cooperation: &mut TabularCooperation<'_>,
    line: &str,
    line_index: usize,
    parse_start_col: usize,
    parse_end_col: Option<usize>,
) -> BuiltinResult<Vec<ParsedNumber>> {
    let mut values = Vec::new();
    for (col_index, raw_field) in line.split(',').enumerate() {
        if col_index > 0 && col_index % 1024 == 0 {
            cooperation.completed(1024).await?;
        }
        if col_index < parse_start_col {
            // Respect csvread(..., row, col) offsets by skipping validation for
            // columns that will be dropped before materializing the output.
            values.push(ParsedNumber::ZERO);
            continue;
        }
        if parse_end_col.is_some_and(|end_col| col_index > end_col) {
            break;
        }
        let trimmed = raw_field.trim();
        if trimmed.is_empty() {
            values.push(ParsedNumber::ZERO);
            continue;
        }
        let unwrapped = if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2
        {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };
        let lowered = unwrapped.to_ascii_lowercase();
        let value = match lowered.as_str() {
            "nan" => ParsedNumber::real(f64::NAN),
            "inf" | "+inf" => ParsedNumber::real(f64::INFINITY),
            "-inf" => ParsedNumber::real(f64::NEG_INFINITY),
            _ => parse_csv_number(unwrapped).ok_or_else(|| {
                csvread_error_with(
                    &CSVREAD_ERROR_NON_NUMERIC_TOKEN,
                    format!(
                        "csvread: nonnumeric token '{}' at row {} column {}",
                        unwrapped,
                        line_index,
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
