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
        selected.check().map_err(|source| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                "writematrix: captured filesystem authority was revoked",
                source,
            )
        })?;
        for col in 0..cols {
            if col > 0 {
                file.write_all(delimiter.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        &WRITEMATRIX_ERROR_IO,
                        format!("writematrix: failed to write delimiter ({err})"),
                        err,
                    )
                })?;
                bytes_written += delimiter.len();
                cooperation.completed(delimiter.len()).await?;
            }
            let cell = matrix.format_cell(row, col, options, &delimiter);
            if !cell.is_empty() {
                file.write_all(cell.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        &WRITEMATRIX_ERROR_IO,
                        format!("writematrix: failed to write value ({err})"),
                        err,
                    )
                })?;
                bytes_written += cell.len();
                cooperation.completed(cell.len()).await?;
            }
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                format!("writematrix: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
        cooperation.completed(line_ending.len()).await?;
    }

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
