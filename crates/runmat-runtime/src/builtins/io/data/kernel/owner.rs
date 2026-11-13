//! Dataset owner domain helpers.

use super::*;

pub(in crate::builtins::io::data) fn dataset_owner(
    runtime: Option<&crate::context::RuntimeContext>,
    base: &Value,
) -> BuiltinResult<AuthorizedHandle> {
    authority::resolve_object(runtime, base, HandleRole::Dataset)
}

pub(in crate::builtins::io::data) fn as_object<'a>(
    value: &'a Value,
    context: &str,
) -> BuiltinResult<&'a ObjectInstance> {
    match value {
        Value::Object(obj) => Ok(obj),
        _ => Err(data_error(format!("{context}: expected object receiver"))),
    }
}

pub(in crate::builtins::io::data) async fn hydrate_dataset_descriptor_on(
    runtime: Option<&crate::context::RuntimeContext>,
    provider: &dyn runmat_filesystem::FsProvider,
    path: &std::path::Path,
    dataset: crate::execution::RootedValue,
) -> BuiltinResult<crate::execution::RootedValue> {
    let request = runmat_filesystem::data_contract::DataManifestRequest {
        path: path.to_string_lossy().into_owned(),
        version: None,
    };
    let descriptor = match provider.data_manifest_descriptor(&request).await {
        Ok(descriptor) => descriptor,
        Err(_) => return Ok(dataset),
    };
    // Roots precede the private payload, so every early error retires that
    // payload before its last original GC registration.
    let (_roots, mut value) = {
        let (value, roots) = dataset.into_value_and_lease();
        (roots, value)
    };
    let Value::Object(obj) = &mut value else {
        return Err(data_error("dataset descriptor receiver is not an object"));
    };
    let fields = [
        (!descriptor.dataset_id.is_empty())
            .then(|| ("__data_id", Value::string(descriptor.dataset_id))),
        Some((
            "__data_version",
            Value::string(format!(
                "{}:{}",
                descriptor.updated_at, descriptor.txn_sequence
            )),
        )),
    ];
    crate::execution::object::replacement::replace_properties(
        runtime,
        obj,
        fields
            .iter()
            .filter_map(|field| field.as_ref().map(|(name, value)| (*name, value))),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:data:HostAllocation",
                message,
                source,
            )
        },
        data_error,
    )?;
    crate::execution::RootedValue::in_optional_context(value, runtime)
}

pub(in crate::builtins::io::data) fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(in crate::builtins::io::data) async fn copy_file_on(
    provider: &dyn runmat_filesystem::FsProvider,
    src: &std::path::Path,
    dst: &std::path::Path,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let bytes = provider.read(src).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to open '{}': {err}", src.display()),
            err,
        )
    })?;
    crate::data::cooperate_data_work(budget, bytes.len() as u64).await?;
    let parent = dst.parent().ok_or_else(|| {
        data_error(format!(
            "invalid destination path '{}': missing parent",
            dst.display()
        ))
    })?;
    provider.create_dir_all(parent).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to create '{}': {err}", parent.display()),
            err,
        )
    })?;
    provider.write(dst, &bytes).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!(
                "failed to copy '{}' -> '{}': {err}",
                src.display(),
                dst.display()
            ),
            err,
        )
    })?;
    crate::data::cooperate_data_work(budget, bytes.len() as u64).await?;
    Ok(())
}

pub(in crate::builtins::io::data) fn make_rel_data_path(
    root: &std::path::Path,
    payload_path: &std::path::Path,
) -> BuiltinResult<String> {
    crate::data::relative_storage_path(root, payload_path)
}

#[async_recursion::async_recursion(?Send)]
pub(in crate::builtins::io::data) async fn copy_dir_recursive_on(
    provider: &dyn runmat_filesystem::FsProvider,
    src: &std::path::Path,
    dst: &std::path::Path,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let metadata = provider.metadata(src).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to stat '{}': {err}", src.display()),
            err,
        )
    })?;
    if !metadata.is_dir() {
        return Err(data_error(format!(
            "expected dataset directory at '{}'",
            src.display()
        )));
    }
    provider.create_dir_all(dst).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to create '{}': {err}", dst.display()),
            err,
        )
    })?;
    for entry in provider.read_dir(src).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to read '{}': {err}", src.display()),
            err,
        )
    })? {
        crate::data::cooperate_data_work(budget, 1).await?;
        let entry_src = entry.path().to_path_buf();
        let entry_dst = dst.join(entry.file_name());
        if entry.is_dir() {
            let own_metadata = provider.symlink_metadata(&entry_src).await.map_err(|err| {
                crate::data::data_error_with_source(
                    format!(
                        "failed to inspect directory entry '{}': {err}",
                        entry_src.display()
                    ),
                    err,
                )
            })?;
            if own_metadata.is_symlink() {
                return Err(data_error(format!(
                    "refusing to recursively copy symbolic-link directory '{}'",
                    entry_src.display()
                )));
            }
            copy_dir_recursive_on(provider, &entry_src, &entry_dst, budget).await?;
            continue;
        }
        copy_file_on(provider, &entry_src, &entry_dst, budget).await?;
    }
    Ok(())
}
