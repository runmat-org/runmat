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
        write_dlm_bytes(
            &mut file,
            cooperation,
            selected,
            newline.as_bytes(),
            &mut bytes,
        )
        .await?;
    }

    for _ in 0..options.roffset {
        selected.check().map_err(|source| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                "dlmwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        write_blank_row(
            cooperation,
            &mut file,
            selected,
            layout.output_cols,
            &options.delimiter,
            newline,
            &mut bytes,
        )
        .await?;
    }

    if rows == 0 || cols == 0 {
        check_dlm_owner(selected)?;
        await_io(runtime, BUILTIN_NAME, file.flush_async())
            .await?
            .map_err(|err| {
                dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    format!("dlmwrite: failed to flush output ({err})"),
                    err,
                )
            })?;
        check_dlm_owner(selected)?;
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
        for field in 0..layout.output_cols {
            if field > 0 {
                write_dlm_bytes(
                    &mut file,
                    cooperation,
                    selected,
                    options.delimiter.as_bytes(),
                    &mut bytes,
                )
                .await?;
            }
            if field >= options.coffset {
                let col = field - options.coffset;
                let idx = row + col * rows;
                let rendered = format_tensor_value(tensor, idx, options)?;
                write_dlm_bytes(
                    &mut file,
                    cooperation,
                    selected,
                    rendered.as_bytes(),
                    &mut bytes,
                )
                .await?;
            }
        }
        write_dlm_bytes(
            &mut file,
            cooperation,
            selected,
            newline.as_bytes(),
            &mut bytes,
        )
        .await?;
    }

    check_dlm_owner(selected)?;
    await_io(runtime, BUILTIN_NAME, file.flush_async())
        .await?
        .map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to flush output ({err})"),
                err,
            )
        })?;
    check_dlm_owner(selected)?;
    Ok(bytes)
}

pub(super) async fn write_blank_row(
    cooperation: &mut TabularCooperation<'_>,
    file: &mut File,
    selected: &TabularFsPath,
    field_count: usize,
    delimiter: &str,
    newline: &str,
    bytes: &mut usize,
) -> BuiltinResult<()> {
    if field_count == 0 {
        write_dlm_bytes(file, cooperation, selected, newline.as_bytes(), bytes).await?;
        return Ok(());
    }
    for _ in 1..field_count {
        write_dlm_bytes(file, cooperation, selected, delimiter.as_bytes(), bytes).await?;
    }
    write_dlm_bytes(file, cooperation, selected, newline.as_bytes(), bytes).await
}

fn check_dlm_owner(selected: &TabularFsPath) -> BuiltinResult<()> {
    selected.check().map_err(|source| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            "dlmwrite: captured filesystem authority was revoked",
            source,
        )
    })
}

async fn write_dlm_bytes(
    file: &mut File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    bytes: &[u8],
    total: &mut usize,
) -> BuiltinResult<()> {
    for chunk in bytes.chunks(4096) {
        (*total).checked_add(chunk.len()).ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_SIZE,
                "dlmwrite: byte count overflow while writing output",
            )
        })?;
        let mut remaining = chunk;
        while !remaining.is_empty() {
            check_dlm_owner(selected)?;
            let count = file.write(remaining).map_err(|source| {
                dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    format!("dlmwrite: failed to write output ({source})"),
                    source,
                )
            })?;
            if count == 0 {
                let source = io::Error::new(
                    io::ErrorKind::WriteZero,
                    "dlmwrite: provider made no progress writing output",
                );
                return Err(dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            add_written_bytes(total, count)?;
            remaining = &remaining[count..];
            cooperation.completed(count).await?;
        }
    }
    Ok(())
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
