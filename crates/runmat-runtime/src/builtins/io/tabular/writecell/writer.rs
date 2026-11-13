use super::*;

pub(super) async fn write_delimited_cells(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    table: &CellTable,
    options: &WriteCellOptions,
) -> BuiltinResult<usize> {
    let path = selected.path();
    let delimiter = options.resolve_delimiter(path);
    let line_ending = options.line_ending.as_str();
    let write_lock = await_io(
        runtime,
        BUILTIN_NAME,
        write_owner::write_lock_for_path(selected),
    )
    .await?
    .map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: write authority unavailable",
            source,
        )
    })?;
    let _write_guard = await_io(runtime, BUILTIN_NAME, write_lock.lock()).await?;

    if options.write_mode == WriteMode::Overwrite {
        return write_staged_delimited(
            runtime,
            cooperation,
            selected,
            table,
            options,
            &delimiter,
            line_ending,
        )
        .await;
    }

    let mut open_options = OpenOptions::new();
    open_options.create(true).write(true).append(true);

    let mut file = await_io(runtime, BUILTIN_NAME, selected.open_with(&open_options))
        .await?
        .map_err(|err| {
            writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                format!(
                    "writecell: unable to open \"{}\" for writing ({err})",
                    path.display()
                ),
                err,
            )
        })?;

    let mut bytes_written = 0usize;
    if await_io(runtime, BUILTIN_NAME, append_needs_line_ending(selected)).await?? {
        write_append_bytes(
            &mut file,
            cooperation,
            selected,
            line_ending.as_bytes(),
            &mut bytes_written,
        )
        .await?;
    }
    for row in 0..table.rows {
        for col in 0..table.cols {
            if col > 0 {
                write_append_bytes(
                    &mut file,
                    cooperation,
                    selected,
                    delimiter.as_bytes(),
                    &mut bytes_written,
                )
                .await?;
            }
            write_append_cell(
                &mut file,
                cooperation,
                selected,
                table.get(row, col),
                options,
                &delimiter,
                &mut bytes_written,
            )
            .await?;
        }
        write_append_bytes(
            &mut file,
            cooperation,
            selected,
            line_ending.as_bytes(),
            &mut bytes_written,
        )
        .await?;
    }
    selected.check().map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: captured filesystem authority was revoked",
            source,
        )
    })?;
    await_io(runtime, BUILTIN_NAME, file.flush_async())
        .await?
        .map_err(|err| {
            writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                format!("writecell: failed to flush output ({err})"),
                err,
            )
        })?;
    selected.check().map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok(bytes_written)
}

async fn write_append_cell(
    file: &mut runmat_filesystem::File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    cell: &CellValue,
    options: &WriteCellOptions,
    delimiter: &str,
    total: &mut usize,
) -> BuiltinResult<()> {
    if let CellValue::Text(text) = cell {
        if !options.quote_strings {
            return write_append_bytes(file, cooperation, selected, text.as_bytes(), total).await;
        }
        write_append_bytes(file, cooperation, selected, b"\"", total).await?;
        for chunk in text.as_bytes().chunks(4096) {
            let mut start = 0usize;
            for (index, byte) in chunk.iter().enumerate() {
                if *byte == b'"' {
                    write_append_bytes(file, cooperation, selected, &chunk[start..index], total)
                        .await?;
                    write_append_bytes(file, cooperation, selected, b"\"\"", total).await?;
                    start = index + 1;
                }
            }
            write_append_bytes(file, cooperation, selected, &chunk[start..], total).await?;
        }
        return write_append_bytes(file, cooperation, selected, b"\"", total).await;
    }
    let rendered = format_cell_for_text(cell, options, delimiter);
    write_append_bytes(file, cooperation, selected, rendered.as_bytes(), total).await
}

async fn write_append_bytes(
    file: &mut runmat_filesystem::File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    bytes: &[u8],
    total: &mut usize,
) -> BuiltinResult<()> {
    for chunk in bytes.chunks(4096) {
        (*total).checked_add(chunk.len()).ok_or_else(|| {
            writecell_error_with(
                &WRITECELL_ERROR_IO,
                "writecell: output byte count exceeds platform limits",
            )
        })?;
        let mut remaining = chunk;
        while !remaining.is_empty() {
            selected.check().map_err(|source| {
                writecell_error_with_source(
                    &WRITECELL_ERROR_IO,
                    "writecell: captured filesystem authority was revoked",
                    source,
                )
            })?;
            let count = match file.write(remaining) {
                Ok(count) => count,
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {
                    cooperation.completed(1).await?;
                    continue;
                }
                Err(source) => {
                    return Err(writecell_error_with_source(
                        &WRITECELL_ERROR_IO,
                        format!("writecell: failed to write delimited text ({source})"),
                        source,
                    ));
                }
            };
            if count > remaining.len() {
                let source = std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "writecell: provider reported more bytes than requested",
                );
                return Err(writecell_error_with_source(
                    &WRITECELL_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            if count == 0 {
                let source = std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "writecell: provider made no progress writing output",
                );
                return Err(writecell_error_with_source(
                    &WRITECELL_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            *total += count;
            remaining = &remaining[count..];
            cooperation.completed(count).await?;
        }
    }
    Ok(())
}

async fn write_staged_delimited(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    table: &CellTable,
    options: &WriteCellOptions,
    delimiter: &str,
    line_ending: &str,
) -> BuiltinResult<usize> {
    let mut stage = write_owner::begin_stage(cooperation, selected)
        .await
        .map_err(map_staged_write_error)?;
    let mut bytes_written = 0usize;
    for row in 0..table.rows {
        for col in 0..table.cols {
            if col > 0 {
                write_staged_segment(
                    runtime,
                    cooperation,
                    selected,
                    stage.as_mut(),
                    delimiter.as_bytes(),
                    &mut bytes_written,
                )
                .await?;
            }
            write_staged_cell(
                runtime,
                cooperation,
                selected,
                stage.as_mut(),
                table.get(row, col),
                options,
                delimiter,
                &mut bytes_written,
            )
            .await?;
        }
        write_staged_segment(
            runtime,
            cooperation,
            selected,
            stage.as_mut(),
            line_ending.as_bytes(),
            &mut bytes_written,
        )
        .await?;
    }
    write_owner::finish_stage(runtime, cooperation, BUILTIN_NAME, selected, stage)
        .await
        .map_err(map_staged_write_error)?;
    Ok(bytes_written)
}

async fn write_staged_cell(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    stage: &mut dyn runmat_filesystem::FsStagedOutput,
    cell: &CellValue,
    options: &WriteCellOptions,
    delimiter: &str,
    total: &mut usize,
) -> BuiltinResult<()> {
    if let CellValue::Text(text) = cell {
        if !options.quote_strings {
            return write_staged_segment(
                runtime,
                cooperation,
                selected,
                stage,
                text.as_bytes(),
                total,
            )
            .await;
        }
        write_staged_segment(runtime, cooperation, selected, stage, b"\"", total).await?;
        for chunk in text.as_bytes().chunks(4096) {
            let mut start = 0usize;
            for (index, byte) in chunk.iter().enumerate() {
                if *byte == b'"' {
                    write_staged_segment(
                        runtime,
                        cooperation,
                        selected,
                        stage,
                        &chunk[start..index],
                        total,
                    )
                    .await?;
                    write_staged_segment(runtime, cooperation, selected, stage, b"\"\"", total)
                        .await?;
                    start = index + 1;
                }
            }
            write_staged_segment(
                runtime,
                cooperation,
                selected,
                stage,
                &chunk[start..],
                total,
            )
            .await?;
        }
        return write_staged_segment(runtime, cooperation, selected, stage, b"\"", total).await;
    }
    let rendered = format_cell_for_text(cell, options, delimiter);
    write_staged_segment(
        runtime,
        cooperation,
        selected,
        stage,
        rendered.as_bytes(),
        total,
    )
    .await
}

async fn write_staged_segment(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    stage: &mut dyn runmat_filesystem::FsStagedOutput,
    bytes: &[u8],
    total: &mut usize,
) -> BuiltinResult<()> {
    let next = total.checked_add(bytes.len()).ok_or_else(|| {
        writecell_error_with(
            &WRITECELL_ERROR_IO,
            "writecell: output byte count exceeds platform limits",
        )
    })?;
    write_owner::write_stage_bytes(runtime, cooperation, BUILTIN_NAME, selected, stage, bytes)
        .await
        .map_err(map_staged_write_error)?;
    *total = next;
    Ok(())
}

fn map_staged_write_error(source: write_owner::StagedWriteError) -> RuntimeError {
    match source {
        write_owner::StagedWriteError::Control(error) => error,
        write_owner::StagedWriteError::PublicationInterrupted(error, disposition) => {
            write_owner::publication_interruption_error(error, disposition)
        }
        write_owner::StagedWriteError::Publication(source) => writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: staged publication failed or its effect could not be confirmed",
            source,
        ),
        write_owner::StagedWriteError::Io(source) => writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: failed to replace delimited text file",
            source,
        ),
    }
}

async fn append_needs_line_ending(selected: &TabularFsPath) -> BuiltinResult<bool> {
    let path = selected.path();
    let metadata = match selected.metadata().await {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                format!(
                    "writecell: unable to inspect \"{}\" ({err})",
                    path.display()
                ),
                err,
            ));
        }
    };
    if metadata.is_empty() {
        return Ok(false);
    }
    let mut file = selected.open_read().await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!(
                "writecell: unable to inspect \"{}\" ({err})",
                path.display()
            ),
            err,
        )
    })?;
    file.seek(SeekFrom::End(-1)).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: unable to inspect file ending ({err})"),
            err,
        )
    })?;
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: unable to read file ending ({err})"),
            err,
        )
    })?;
    Ok(!matches!(byte[0], b'\n' | b'\r'))
}

pub(super) async fn write_spreadsheet_cells(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    table: &CellTable,
    options: &WriteCellOptions,
) -> BuiltinResult<usize> {
    if options.write_mode == WriteMode::Append {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: WriteMode 'append' is not supported for spreadsheet files",
        ));
    }
    let range_start = options.range_start();
    let end_row = range_start.row.checked_add(table.rows).ok_or_else(|| {
        writecell_error_with(&WRITECELL_ERROR_OPTION, "writecell: Range row overflow")
    })?;
    let end_col = range_start.col.checked_add(table.cols).ok_or_else(|| {
        writecell_error_with(&WRITECELL_ERROR_OPTION, "writecell: Range column overflow")
    })?;
    if end_row > MAX_EXCEL_ROW_INDEX + 1 || end_col > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Range exceeds Excel worksheet limits",
        ));
    }

    let bytes = build_xlsx_workbook(table, &options.sheet_name(), range_start)?;
    write_owner::safe_replace_file(runtime, cooperation, BUILTIN_NAME, selected, &bytes)
        .await
        .map_err(|source| match source {
            write_owner::StagedWriteError::Control(error) => error,
            write_owner::StagedWriteError::PublicationInterrupted(error, disposition) => {
                write_owner::publication_interruption_error(error, disposition)
            }
            write_owner::StagedWriteError::Publication(source) => writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                "writecell: staged publication failed or its effect could not be confirmed",
                source,
            ),
            write_owner::StagedWriteError::Io(source) => writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                "writecell: failed to replace spreadsheet file",
                source,
            ),
        })?;
    Ok(bytes.len())
}
