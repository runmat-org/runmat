use super::*;

pub(super) async fn write_dlm(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    tensor: &Tensor,
    options: &DlmWriteOptions,
) -> BuiltinResult<usize> {
    let path = selected.path();
    let rows = tensor.rows();
    let cols = tensor.cols();
    let newline = options.newline.as_str();
    let layout = validate_output_layout(rows, cols, options)?;

    let (existing_nonempty, ends_with_newline) = if options.append {
        match await_io(runtime, BUILTIN_NAME, selected.metadata()).await? {
            Ok(meta) if !meta.is_empty() => {
                let ends = await_io(runtime, BUILTIN_NAME, file_ends_with_newline(selected))
                    .await?
                    .map_err(|err| {
                        dlmwrite_error_with_source(
                            &DLMWRITE_ERROR_IO,
                            format!(
                                "dlmwrite: failed to inspect existing file \"{}\" ({err})",
                                path.display()
                            ),
                            err,
                        )
                    })?;
                (true, ends)
            }
            Ok(_) => (false, false),
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    (false, false)
                } else {
                    return Err(dlmwrite_error_with_source(
                        &DLMWRITE_ERROR_IO,
                        format!("dlmwrite: unable to inspect \"{}\" ({err})", path.display()),
                        err,
                    ));
                }
            }
        }
    } else {
        (false, false)
    };

    let mut open = OpenOptions::new();
    open.create(true);
    if options.append {
        open.append(true);
    } else {
        open.write(true).truncate(true);
    }

    let mut file = await_io(runtime, BUILTIN_NAME, selected.open_with(&open))
        .await?
        .map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!(
                    "dlmwrite: unable to open \"{}\" for writing ({err})",
                    path.display()
                ),
                err,
            )
        })?;

    let mut bytes = 0usize;

    if options.append && existing_nonempty && !ends_with_newline {
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to insert newline before append ({err})"),
                err,
            )
        })?;
        add_written_bytes(&mut bytes, newline.len())?;
        cooperation.completed(newline.len()).await?;
    }

    for _ in 0..options.roffset {
        selected.check().map_err(|source| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                "dlmwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        let written = write_blank_row(
            cooperation,
            &mut file,
            layout.output_cols,
            &options.delimiter,
            newline,
        )
        .await?;
        add_written_bytes(&mut bytes, written)?;
    }

    if rows == 0 || cols == 0 {
        await_io(runtime, BUILTIN_NAME, file.flush_async())
            .await?
            .map_err(|err| {
                dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    format!("dlmwrite: failed to flush output ({err})"),
                    err,
                )
            })?;
        return Ok(bytes);
    }

    for row in 0..rows {
        selected.check().map_err(|source| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                "dlmwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        let mut fields = Vec::with_capacity(layout.output_cols);
        for index in 0..options.coffset {
            fields.push(String::new());
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(options.coffset % 1024).await?;
        for col in 0..cols {
            let idx = row + col * rows;
            fields.push(format_tensor_value(tensor, idx, options)?);
            if col % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(cols % 1024).await?;
        let line = fields.join(&options.delimiter);
        if !line.is_empty() {
            file.write_all(line.as_bytes()).map_err(|err| {
                dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    format!("dlmwrite: failed to write data ({err})"),
                    err,
                )
            })?;
            add_written_bytes(&mut bytes, line.len())?;
        }
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write newline ({err})"),
                err,
            )
        })?;
        add_written_bytes(&mut bytes, newline.len())?;
        cooperation.completed(line.len() + newline.len()).await?;
    }

    await_io(runtime, BUILTIN_NAME, file.flush_async())
        .await?
        .map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to flush output ({err})"),
                err,
            )
        })?;
    Ok(bytes)
}

pub(super) async fn write_blank_row(
    cooperation: &mut TabularCooperation<'_>,
    file: &mut File,
    field_count: usize,
    delimiter: &str,
    newline: &str,
) -> BuiltinResult<usize> {
    let mut bytes = 0usize;
    if field_count == 0 {
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write offset newline ({err})"),
                err,
            )
        })?;
        cooperation.completed(newline.len()).await?;
        return Ok(newline.len());
    }
    for field in 1..field_count {
        file.write_all(delimiter.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write offset row ({err})"),
                err,
            )
        })?;
        add_written_bytes(&mut bytes, delimiter.len())?;
        if field % 1024 == 0 {
            cooperation
                .completed(1024usize.saturating_mul(delimiter.len()))
                .await?;
        }
    }
    cooperation
        .completed(((field_count - 1) % 1024).saturating_mul(delimiter.len()))
        .await?;
    file.write_all(newline.as_bytes()).map_err(|err| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            format!("dlmwrite: failed to write offset newline ({err})"),
            err,
        )
    })?;
    add_written_bytes(&mut bytes, newline.len())?;
    cooperation.completed(newline.len()).await?;
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DlmWriteLayout {
    pub(super) output_cols: usize,
}

pub(super) fn validate_output_layout(
    rows: usize,
    cols: usize,
    options: &DlmWriteOptions,
) -> BuiltinResult<DlmWriteLayout> {
    let output_rows = rows.checked_add(options.roffset).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_SIZE,
            "dlmwrite: row offset overflows the output layout",
        )
    })?;
    let output_cols = cols.checked_add(options.coffset).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_SIZE,
            "dlmwrite: column offset overflows the output layout",
        )
    })?;
    let fields = output_rows.checked_mul(output_cols).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_SIZE,
            "dlmwrite: row and column offsets overflow the output layout",
        )
    })?;
    if output_rows > MAX_DLMWRITE_LAYOUT_FIELDS
        || output_cols > MAX_DLMWRITE_LAYOUT_FIELDS
        || fields > MAX_DLMWRITE_LAYOUT_FIELDS
    {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_SIZE,
            format!(
                "dlmwrite: requested output layout exceeds the {MAX_DLMWRITE_LAYOUT_FIELDS}-field safety limit"
            ),
        ));
    }
    Ok(DlmWriteLayout { output_cols })
}

pub(super) fn add_written_bytes(total: &mut usize, written: usize) -> BuiltinResult<()> {
    *total = total.checked_add(written).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_SIZE,
            "dlmwrite: byte count overflow while writing output",
        )
    })?;
    Ok(())
}

pub(super) async fn file_ends_with_newline(selected: &TabularFsPath) -> io::Result<bool> {
    let metadata = selected.metadata().await?;
    let len = metadata.len();
    if len == 0 {
        return Ok(false);
    }
    let mut file = selected.open_read().await?;
    let to_read = len.min(2) as usize;
    file.seek(SeekFrom::End(-(to_read as i64)))?;
    let mut buffer = vec![0u8; to_read];
    file.read_exact(&mut buffer)?;
    Ok(buffer.contains(&b'\n') || buffer.contains(&b'\r'))
}
