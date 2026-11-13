//! Public data path and lifecycle operations.

use super::path_authority::DataPathAuthority;
use super::*;
use crate::context::host_effect::DatasetPathOperation;
use crate::context::RuntimeContext;

mod admission;

#[runtime_builtin(
    name = "data.create",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_create_builtin(
    runtime: Option<RuntimeContext>,
    path: Value,
    schema: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let path = parse_string(&path, "data.create path")?;
    let schema = parse_schema(&schema)?;
    for meta in schema.arrays.values() {
        DataArrayPayload::validate_zeros(&meta.dtype, &meta.shape)?;
    }
    let authority = DataPathAuthority::capture(&[&path])?;
    let root = authority.resolve(&path)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Create,
        &root,
        None,
        admission::facts(&(&path, &schema.arrays)),
    )?;
    let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
    admission::execute(runtime.as_ref(), receipt, async {
        let now = now_rfc3339();
        let mut arrays = BTreeMap::new();
        for (name, mut meta) in schema.arrays {
            crate::data::cooperate_data_work(&mut budget, 0).await?;
            let payload = DataArrayPayload::zeros_cooperative(
                meta.dtype.clone(),
                meta.shape.clone(),
                &mut budget,
            )
            .await?;
            crate::data::cooperate_data_work(&mut budget, 0).await?;
            let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
                authority.provider(),
                &root,
                &name,
                &payload,
                &meta.chunk_shape,
                &mut budget,
            )
            .await?;
            meta.data_path = make_rel_data_path(&root, &payload_path)?;
            meta.chunk_index_path = Some(make_rel_data_path(&root, &chunk_index_path)?);
            arrays.insert(name, meta);
        }

        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".to_string(),
            dataset_id: crate::data::new_dataset_id(),
            name: root.file_name().map(|v| v.to_string_lossy().to_string()),
            created_at: now.clone(),
            updated_at: now,
            arrays,
            attrs: BTreeMap::new(),
            txn_sequence: 0,
        };
        crate::data::write_manifest_on(authority.provider(), &root, &manifest).await?;
        let lease = crate::data::authority::issue_dataset(
            runtime.as_ref(),
            authority.provider_arc().clone(),
            root.clone(),
            path.clone(),
        )?;
        dataset_object(runtime.as_ref(), &path, &manifest, lease)
    })
    .await
}

#[runtime_builtin(
    name = "data.open",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_open_builtin(
    runtime: Option<RuntimeContext>,
    path: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let path = parse_string(&path, "data.open path")?;
    let authority = DataPathAuthority::capture(&[&path])?;
    let root = authority.resolve(&path)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Open,
        &root,
        None,
        admission::facts(&path),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
        let lease = crate::data::authority::issue_dataset(
            runtime.as_ref(),
            authority.provider_arc().clone(),
            root.clone(),
            path.clone(),
        )?;
        let ds = dataset_object(runtime.as_ref(), &path, &manifest, lease)?;
        hydrate_dataset_descriptor_on(runtime.as_ref(), authority.provider(), &root, ds).await
    })
    .await
}

#[runtime_builtin(
    name = "data.exists",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_exists_builtin(runtime: Option<RuntimeContext>, path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.exists path")?;
    let authority = DataPathAuthority::capture(&[&path])?;
    let root = authority.resolve(&path)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Exists,
        &root,
        None,
        admission::facts(&path),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        let exists = authority
            .provider()
            .metadata(&manifest_path(&root))
            .await
            .is_ok();
        Ok(Value::Bool(exists))
    })
    .await
}

#[runtime_builtin(
    name = "data.delete",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_delete_builtin(runtime: Option<RuntimeContext>, path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.delete path")?;
    let authority = DataPathAuthority::capture(&[&path])?;
    let root = authority.resolve(&path)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Delete,
        &root,
        None,
        admission::facts(&path),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        authority
            .provider()
            .remove_dir_all(&root)
            .await
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!("data.delete: failed to remove '{}': {err}", root.display()),
                    err,
                )
            })?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "data.copy",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_copy_builtin(
    runtime: Option<RuntimeContext>,
    from_path: Value,
    to_path: Value,
) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.copy fromPath")?;
    let to = parse_string(&to_path, "data.copy toPath")?;
    let authority = DataPathAuthority::capture(&[&from, &to])?;
    let from_root = authority.resolve(&from)?;
    let to_root = authority.resolve(&to)?;
    authority
        .admit_tree_destination(&from_root, &to_root)
        .await?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Copy,
        &from_root,
        Some(&to_root),
        admission::facts(&(&from, &to)),
    )?;
    let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
    admission::execute(runtime.as_ref(), receipt, async {
        copy_dir_recursive_on(authority.provider(), &from_root, &to_root, &mut budget).await?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "data.move",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_move_builtin(
    runtime: Option<RuntimeContext>,
    from_path: Value,
    to_path: Value,
) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.move fromPath")?;
    let to = parse_string(&to_path, "data.move toPath")?;
    let authority = DataPathAuthority::capture(&[&from, &to])?;
    let from_root = authority.resolve(&from)?;
    let to_root = authority.resolve(&to)?;
    authority
        .admit_tree_destination(&from_root, &to_root)
        .await?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Move,
        &from_root,
        Some(&to_root),
        admission::facts(&(&from, &to)),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        authority
            .provider()
            .rename(&from_root, &to_root)
            .await
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!("data.move: failed to move dataset '{from}' -> '{to}': {err}"),
                    err,
                )
            })?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "data.import",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_import_builtin(
    runtime: Option<RuntimeContext>,
    path: Value,
    format: Value,
    source_path: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let path = parse_string(&path, "data.import path")?;
    let format = parse_string(&format, "data.import format")?;
    if !format.eq_ignore_ascii_case("data") {
        return Err(data_error(
            "data.import currently supports only format='data'",
        ));
    }
    let source_path = parse_string(&source_path, "data.import sourcePath")?;
    let authority = DataPathAuthority::capture(&[&path, &source_path])?;
    let root = authority.resolve(&path)?;
    let source_root = authority.resolve(&source_path)?;
    authority
        .admit_tree_destination(&source_root, &root)
        .await?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Import,
        &source_root,
        Some(&root),
        admission::facts(&(&path, &format, &source_path)),
    )?;
    let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
    admission::execute(runtime.as_ref(), receipt, async {
        copy_dir_recursive_on(authority.provider(), &source_root, &root, &mut budget).await?;
        let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
        let lease = crate::data::authority::issue_dataset(
            runtime.as_ref(),
            authority.provider_arc().clone(),
            root.clone(),
            path.clone(),
        )?;
        dataset_object(runtime.as_ref(), &path, &manifest, lease)
    })
    .await
}

#[runtime_builtin(
    name = "data.export",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_export_builtin(
    runtime: Option<RuntimeContext>,
    path: Value,
    format: Value,
    target_path: Value,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.export path")?;
    let format = parse_string(&format, "data.export format")?;
    if !format.eq_ignore_ascii_case("data") {
        return Err(data_error(
            "data.export currently supports only format='data'",
        ));
    }
    let target_path = parse_string(&target_path, "data.export targetPath")?;
    let authority = DataPathAuthority::capture(&[&path, &target_path])?;
    let source_root = authority.resolve(&path)?;
    let target_root = authority.resolve(&target_path)?;
    authority
        .admit_tree_destination(&source_root, &target_root)
        .await?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Export,
        &source_root,
        Some(&target_root),
        admission::facts(&(&path, &format, &target_path)),
    )?;
    let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
    admission::execute(runtime.as_ref(), receipt, async {
        copy_dir_recursive_on(
            authority.provider(),
            &source_root,
            &target_root,
            &mut budget,
        )
        .await?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "data.list",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_list_builtin(
    runtime: Option<RuntimeContext>,
    path_prefix: Value,
) -> BuiltinResult<Value> {
    let prefix = parse_string(&path_prefix, "data.list prefix")?;
    let authority = DataPathAuthority::capture(&[&prefix])?;
    let root = authority.resolve(&prefix)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::List,
        &root,
        None,
        admission::facts(&prefix),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        let entries = authority.provider().read_dir(&root).await.map_err(|err| {
            crate::data::data_error_with_source(
                format!("data.list: failed to read '{}': {err}", root.display()),
                err,
            )
        })?;
        let mut values = Vec::new();
        for entry in entries {
            if !entry.is_dir() {
                continue;
            }
            let candidate = entry.path();
            if candidate.extension().and_then(|s| s.to_str()) != Some("data") {
                continue;
            }
            if authority
                .provider()
                .metadata(&candidate.join("manifest.json"))
                .await
                .is_ok()
            {
                values.push(Value::string(
                    PathBuf::from(&prefix)
                        .join(entry.file_name())
                        .to_string_lossy()
                        .into_owned(),
                ));
            }
        }
        let cols = values.len();
        make_cell(values, 1, cols).map_err(data_error)
    })
    .await
}

#[runtime_builtin(
    name = "data.inspect",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::path"
)]
async fn data_inspect_builtin(
    runtime: Option<RuntimeContext>,
    path: Value,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.inspect path")?;
    let authority = DataPathAuthority::capture(&[&path])?;
    let root = authority.resolve(&path)?;
    let receipt = admission::select(
        runtime.as_ref(),
        &authority,
        DatasetPathOperation::Inspect,
        &root,
        None,
        admission::facts(&path),
    )?;
    admission::execute(runtime.as_ref(), receipt, async {
        let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
        let mut out = runmat_value::UnboundedStructValueBuilder::new();
        out.fields.insert("path".to_string(), Value::string(path));
        out.fields
            .insert("id".to_string(), Value::string(manifest.dataset_id));
        out.fields.insert(
            "arrayCount".to_string(),
            Value::Num(manifest.arrays.len() as f64),
        );
        out.fields
            .insert("updatedAt".to_string(), Value::string(manifest.updated_at));
        Ok(Value::Struct(out.finish()))
    })
    .await
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "path/tests.rs"]
mod tests;
