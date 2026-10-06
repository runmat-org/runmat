use super::*;

pub(super) async fn write_bytes(
    path: &Path,
    payload: &[u8],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let temp_path = temporary_output_path(path);
    let mut created_temp = false;
    let result: BuiltinResult<()> = async {
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        let mut file = options.open_async(&temp_path).await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to open temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        created_temp = true;
        file.write_all(payload).map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to write temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        file.flush_async().await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to flush temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        file.sync_all_async().await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to sync temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        drop(file);
        runmat_filesystem::rename_async(&temp_path, path)
            .await
            .map_err(|err| {
                persistence_error_with_source(
                    &ERROR_IO,
                    builtin,
                    format!(
                        "{builtin}: unable to replace '{}' with temporary file '{}'",
                        path.display(),
                        temp_path.display()
                    ),
                    err,
                )
            })?;
        Ok(())
    }
    .await;
    if result.is_err() && created_temp {
        let _ = runmat_filesystem::remove_file_async(&temp_path).await;
    }
    result
}

pub(super) fn temporary_output_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("runmat-figure"));
    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(
        ".{}.tmp",
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    path.with_file_name(temp_name)
}

pub(super) fn persistence_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn persistence_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
