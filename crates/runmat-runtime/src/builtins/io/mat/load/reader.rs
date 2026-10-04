use super::*;

pub(crate) async fn read_mat_file_for_builtin(
    path: &Path,
    builtin: &str,
) -> crate::BuiltinResult<Vec<(String, Value)>> {
    match read_mat_file(path).await {
        Ok(entries) => Ok(entries),
        Err(err) => {
            let message = err.message().replacen("load:", &format!("{builtin}:"), 1);
            let mut builder = build_runtime_error(message).with_builtin(builtin);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier);
            }
            Err(builder.with_source(err).build())
        }
    }
}

pub(crate) async fn read_mat_file_for_builtin_on_with_resource_liveness(
    source: &MatFsPath,
    builtin: &str,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> crate::BuiltinResult<Vec<(String, Value)>> {
    match read_mat_file_on_with_resource_liveness(source, runtime, resource_liveness).await {
        Ok(entries) => Ok(entries),
        Err(err) => {
            let message = err.message().replacen("load:", &format!("{builtin}:"), 1);
            let mut builder = build_runtime_error(message).with_builtin(builtin);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier);
            }
            Err(builder.with_source(err).build())
        }
    }
}

pub(crate) async fn read_mat_file(path: &Path) -> BuiltinResult<Vec<(String, Value)>> {
    let source = MatFsPath::capture(path.to_path_buf()).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    read_mat_file_on(&source).await
}

pub(crate) async fn read_mat_file_on(source: &MatFsPath) -> BuiltinResult<Vec<(String, Value)>> {
    read_mat_file_on_with_context(source, None).await
}

pub(crate) async fn read_mat_file_on_with_context(
    source: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<(String, Value)>> {
    read_mat_file_on_with_resource_liveness(source, runtime, None).await
}

pub(crate) async fn read_mat_file_on_with_resource_liveness(
    source: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cooperation =
        decoder::MatCodecCooperation::new(runtime, Some(source), "load", &LOAD_ERROR_IO)
            .with_resource_liveness(resource_liveness);
    let file = cooperation.await_io(source.open()).await?.map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to open '{}': {err}", source.path().display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    let entries = decoder::read_mat_reader_cooperative(&mut reader, &mut cooperation).await?;
    check_source_current(source)?;
    restore_loaded_entries(entries, Some(source), Some(&mut cooperation)).await
}

pub fn decode_workspace_from_mat_bytes(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cursor = Cursor::new(bytes);
    let entries = read_mat_reader(&mut cursor)?;
    futures::executor::block_on(restore_loaded_entries(entries, None, None))
}

pub(super) async fn restore_loaded_entries(
    entries: Vec<(String, Value)>,
    source: Option<&MatFsPath>,
    mut cooperation: Option<&mut decoder::MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut restored = Vec::new();
    restored.try_reserve_exact(entries.len()).map_err(|error| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            "load: restored values cannot be allocated",
            error,
        )
    })?;
    for (name, value) in entries {
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(0).await?;
        }
        if let Some(source) = source {
            check_source_current(source)?;
        }
        let value = match cooperation.as_deref_mut() {
            Some(cooperation) => {
                cooperation
                    .await_io(restore_value_from_mat_load(value))
                    .await??
            }
            None => restore_value_from_mat_load(value).await?,
        };
        restored.push((name, value));
        if let Some(source) = source {
            check_source_current(source)?;
        }
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(1).await?;
        }
    }
    if let Some(cooperation) = cooperation {
        cooperation.completed(0).await?;
    }
    Ok(restored)
}

fn check_source_current(source: &MatFsPath) -> BuiltinResult<()> {
    source.check_current().map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!(
                "load: MAT source authority for '{}' changed: {err}",
                source.path().display()
            ),
            err,
        )
    })
}
