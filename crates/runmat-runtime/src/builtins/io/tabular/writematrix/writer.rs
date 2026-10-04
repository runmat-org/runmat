use super::*;
use runmat_filesystem::OpenOptions;
use std::io::Write;

pub(super) async fn write_matrix(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    matrix: &MatrixData,
    options: &WriteMatrixOptions,
) -> BuiltinResult<usize> {
    let path = selected.path();
    let delimiter = options.resolve_delimiter(path);
    let line_ending = options.line_ending.as_str();

    let mut open_options = OpenOptions::new();
    open_options.create(true);
    match options.write_mode {
        WriteMode::Overwrite => {
            open_options.write(true).truncate(true);
        }
        WriteMode::Append => {
            open_options.write(true).append(true);
        }
    }

    let mut file = await_io(runtime, BUILTIN_NAME, selected.open_with(&open_options))
        .await?
        .map_err(|err| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                format!(
                    "writematrix: unable to open \"{}\" for writing ({err})",
                    path.display()
                ),
                err,
            )
        })?;

    let mut bytes_written = 0usize;
    let rows = matrix.rows();
    let cols = matrix.cols();

    if rows == 0 {
        // Nothing else to do; file was truncated or created above.
        check_owner(selected)?;
        await_io(runtime, BUILTIN_NAME, file.flush_async())
            .await?
            .map_err(|err| {
                writematrix_error_with_source(
                    &WRITEMATRIX_ERROR_IO,
                    format!("writematrix: failed to flush output ({err})"),
                    err,
                )
            })?;
        selected.check().map_err(|source| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                "writematrix: captured filesystem authority was revoked",
                source,
            )
        })?;
        return Ok(0);
    }

    for row in 0..rows {
        for col in 0..cols {
            if col > 0 {
                write_bounded(
                    &mut file,
                    cooperation,
                    selected,
                    delimiter.as_bytes(),
                    &mut bytes_written,
                )
                .await?;
            }
            match matrix.cell_at(row, col, options) {
                MatrixCell::Numeric(cell) => {
                    write_bounded(
                        &mut file,
                        cooperation,
                        selected,
                        cell.as_bytes(),
                        &mut bytes_written,
                    )
                    .await?;
                }
                MatrixCell::Text(cell) => {
                    write_text_cell(
                        &mut file,
                        cooperation,
                        selected,
                        cell,
                        options.quote_strings,
                        &delimiter,
                        &mut bytes_written,
                    )
                    .await?;
                }
            }
        }
        write_bounded(
            &mut file,
            cooperation,
            selected,
            line_ending.as_bytes(),
            &mut bytes_written,
        )
        .await?;
    }

    check_owner(selected)?;
    await_io(runtime, BUILTIN_NAME, file.flush_async())
        .await?
        .map_err(|err| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                format!("writematrix: failed to flush output ({err})"),
                err,
            )
        })?;

    selected.check().map_err(|source| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            "writematrix: captured filesystem authority was revoked",
            source,
        )
    })?;

    Ok(bytes_written)
}

fn check_owner(selected: &TabularFsPath) -> BuiltinResult<()> {
    selected.check().map_err(|source| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            "writematrix: captured filesystem authority was revoked",
            source,
        )
    })
}

async fn write_bounded(
    file: &mut runmat_filesystem::File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    bytes: &[u8],
    written: &mut usize,
) -> BuiltinResult<()> {
    for chunk in bytes.chunks(4096) {
        (*written).checked_add(chunk.len()).ok_or_else(|| {
            writematrix_error_with(
                &WRITEMATRIX_ERROR_IO,
                "writematrix: output byte count exceeds platform limits",
            )
        })?;
        let mut remaining = chunk;
        while !remaining.is_empty() {
            check_owner(selected)?;
            let count = match file.write(remaining) {
                Ok(count) => count,
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {
                    cooperation.completed(1).await?;
                    continue;
                }
                Err(source) => {
                    return Err(writematrix_error_with_source(
                        &WRITEMATRIX_ERROR_IO,
                        format!("writematrix: failed to write output ({source})"),
                        source,
                    ));
                }
            };
            if count > remaining.len() {
                let source = std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "writematrix: provider reported more bytes than requested",
                );
                return Err(writematrix_error_with_source(
                    &WRITEMATRIX_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            if count == 0 {
                let source = std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "writematrix: provider made no progress writing output",
                );
                return Err(writematrix_error_with_source(
                    &WRITEMATRIX_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            *written += count;
            remaining = &remaining[count..];
            cooperation.completed(count).await?;
        }
    }
    Ok(())
}

async fn write_text_cell(
    file: &mut runmat_filesystem::File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    value: &str,
    quote: QuoteStrings,
    delimiter: &str,
    written: &mut usize,
) -> BuiltinResult<()> {
    if !string_needs_quote(value, quote, delimiter, cooperation).await? {
        return write_bounded(file, cooperation, selected, value.as_bytes(), written).await;
    }
    write_bounded(file, cooperation, selected, b"\"", written).await?;
    for chunk in value.as_bytes().chunks(4096) {
        let mut start = 0usize;
        for (index, byte) in chunk.iter().enumerate() {
            if *byte == b'"' {
                write_bounded(file, cooperation, selected, &chunk[start..index], written).await?;
                write_bounded(file, cooperation, selected, b"\"\"", written).await?;
                start = index + 1;
            }
        }
        write_bounded(file, cooperation, selected, &chunk[start..], written).await?;
    }
    write_bounded(file, cooperation, selected, b"\"", written).await
}

async fn string_needs_quote(
    value: &str,
    quote: QuoteStrings,
    delimiter: &str,
    cooperation: &mut TabularCooperation<'_>,
) -> BuiltinResult<bool> {
    match quote {
        QuoteStrings::All => return Ok(true),
        QuoteStrings::None => return Ok(false),
        QuoteStrings::Minimal => {}
    }
    let bytes = value.as_bytes();
    let needle = delimiter.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let plain_end = start.saturating_add(4096).min(bytes.len());
        let overlap_end = plain_end
            .saturating_add(needle.len().saturating_sub(1))
            .min(bytes.len());
        let plain = &bytes[start..plain_end];
        if plain
            .iter()
            .any(|byte| matches!(*byte, b'"' | b'\n' | b'\r'))
            || (!needle.is_empty()
                && bytes[start..overlap_end]
                    .windows(needle.len())
                    .any(|window| window == needle))
        {
            return Ok(true);
        }
        cooperation.completed(plain.len()).await?;
        start = plain_end;
    }
    Ok(false)
}
