use super::*;

mod read_set;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod retained_tests;
pub(super) use read_set::WorkspaceReadSet;

pub(super) async fn ensure_workspace_entries<'a>(
    cache: &'a mut Option<Vec<(String, Value)>>,
    workspace: &mut WorkspaceReadSet<'_>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<&'a Vec<(String, Value)>> {
    if cache.is_none() {
        let entries = collect_workspace_entries(workspace, cooperation).await?;
        *cache = Some(entries);
    }
    Ok(cache.as_ref().unwrap())
}

pub(super) async fn collect_workspace_entries(
    workspace: &mut WorkspaceReadSet<'_>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let snapshot = workspace
        .snapshot()?
        .ok_or_else(|| save_error_with(&SAVE_ERROR_WORKSPACE, SAVE_ERROR_WORKSPACE.message))?;
    workspace.retain_values(snapshot.iter().map(|(_, value)| value))?;
    cooperation.completed(0).await?;
    let mut entries = Vec::with_capacity(snapshot.len());
    for (name, value) in snapshot {
        let gathered = cooperation
            .await_io(gather_if_needed_async(&value))
            .await??;
        entries.push((name, gathered));
    }
    workspace.retain_values(entries.iter().map(|(_, value)| value))?;
    Ok(entries)
}

pub(super) fn find_in_entries(entries: &[(String, Value)], name: &str) -> Option<Value> {
    entries
        .iter()
        .find(|(entry_name, _)| entry_name == name)
        .map(|(_, value)| value.clone())
}

pub(super) fn deduplicate_entries(entries: Vec<(String, Value)>) -> Vec<(String, Value)> {
    let mut seen = HashSet::new();
    let mut unique_entries = Vec::new();
    for (name, value) in entries.into_iter().rev() {
        if seen.insert(name.clone()) {
            unique_entries.push((name, value));
        }
    }
    unique_entries.reverse();
    unique_entries
}

pub(super) async fn append_existing_entries(
    path: &MatFsPath,
    new_entries: Vec<(String, Value)>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut MatCodecCooperation<'_>,
    workspace: &mut WorkspaceReadSet<'_>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let entries = read_existing_entries_for_append(path, runtime, cooperation).await?;
    entries.with_owned(|entries| {
        workspace.retain_values(entries.0.iter().map(|(_, value)| value))?;
        let mut entries = entries.0;
        entries.extend(new_entries);
        Ok(deduplicate_entries(entries))
    })
}

async fn read_existing_entries_for_append(
    path: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<super::super::load::RootedMatEntries> {
    match cooperation.await_io(path.metadata()).await? {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return super::super::load::RootedMatEntries::new(super::super::load::MatEntries(
                Vec::new(),
            ))
            .map_err(|error| error.into_runtime_error());
        }
        Err(err) => {
            return Err(save_error_with_source(
                &SAVE_ERROR_IO,
                format!(
                    "save: failed to inspect existing MAT-file '{}': {err}",
                    path.path().display()
                ),
                err,
            ));
        }
    }
    match read_mat_file_on_with_resource_liveness(
        path,
        runtime,
        cooperation.resource_liveness_lease(),
    )
    .await
    {
        Ok(entries) => Ok(entries),
        Err(err) => Err(save_error_with_source(
            &SAVE_ERROR_IO,
            format!(
                "save: failed to read existing MAT-file '{}': {}",
                path.path().display(),
                err.message()
            ),
            err,
        )),
    }
}

pub(super) async fn append_struct_fields(
    struct_name: &str,
    value: &StructValue,
    fields: &Option<Vec<String>>,
    out: &mut Vec<(String, Value)>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<()> {
    if let Some(ref names) = fields {
        for field in names {
            match value.fields.get(field) {
                Some(val) => {
                    let gathered = cooperation.await_io(gather_if_needed_async(val)).await??;
                    out.push((field.clone(), gathered));
                }
                None => {
                    return Err(save_error_with(
                        &SAVE_ERROR_SELECTION,
                        format!(
                            "save: struct '{}' does not have a field named '{}'",
                            struct_name, field
                        ),
                    ));
                }
            }
        }
    } else {
        for (field, val) in &value.fields {
            let gathered = cooperation.await_io(gather_if_needed_async(val)).await??;
            out.push((field.to_owned(), gathered));
        }
    }
    Ok(())
}

pub(super) async fn lookup_workspace(
    workspace: &mut WorkspaceReadSet<'_>,
    name: &str,
) -> BuiltinResult<Value> {
    let value = workspace.lookup(name)?.ok_or_else(|| {
        save_error_with(
            &SAVE_ERROR_SELECTION,
            format!("save: variable '{}' was not found in the workspace", name),
        )
    })?;
    workspace.retain_values(std::iter::once(&value))?;
    let gathered = gather_if_needed_async(&value).await?;
    workspace.retain_values(std::iter::once(&gathered))?;
    Ok(gathered)
}
