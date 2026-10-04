//! Public data path and lifecycle operations.

use super::path_authority::DataPathAuthority;
use super::*;

#[runtime_builtin(
    name = "data.create",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_create_builtin(path: Value, schema: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.create path")?;
    let schema = parse_schema(&schema)?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    let now = now_rfc3339();
    let mut arrays = BTreeMap::new();
    for (name, mut meta) in schema.arrays {
        let payload = DataArrayPayload::zeros(meta.dtype.clone(), meta.shape.clone())?;
        let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
            authority.provider(),
            &root,
            &name,
            &payload,
            &meta.chunk_shape,
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
    Ok(dataset_object(&path, &manifest))
}

#[runtime_builtin(
    name = "data.open",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_open_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.open path")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
    let mut ds = dataset_object(&path, &manifest);
    hydrate_dataset_descriptor_on(authority.provider(), &root, &mut ds).await;
    Ok(ds)
}

#[runtime_builtin(
    name = "data.exists",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_exists_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.exists path")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    let exists = authority
        .provider()
        .metadata(&manifest_path(&root))
        .await
        .is_ok();
    Ok(Value::Bool(exists))
}

#[runtime_builtin(
    name = "data.delete",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_delete_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.delete path")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    authority
        .provider()
        .remove_dir_all(&root)
        .await
        .map_err(|err| {
            data_error(format!(
                "data.delete: failed to remove '{}': {err}",
                root.display()
            ))
        })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.copy",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_copy_builtin(from_path: Value, to_path: Value) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.copy fromPath")?;
    let to = parse_string(&to_path, "data.copy toPath")?;
    let authority = DataPathAuthority::capture()?;
    copy_dir_recursive_on(
        authority.provider(),
        &authority.resolve(&from),
        &authority.resolve(&to),
    )
    .await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.move",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_move_builtin(from_path: Value, to_path: Value) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.move fromPath")?;
    let to = parse_string(&to_path, "data.move toPath")?;
    let authority = DataPathAuthority::capture()?;
    authority
        .provider()
        .rename(&authority.resolve(&from), &authority.resolve(&to))
        .await
        .map_err(|err| {
            data_error(format!(
                "data.move: failed to move dataset '{from}' -> '{to}': {err}"
            ))
        })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.import",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_import_builtin(
    path: Value,
    format: Value,
    source_path: Value,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.import path")?;
    let format = parse_string(&format, "data.import format")?;
    if !format.eq_ignore_ascii_case("data") {
        return Err(data_error(
            "data.import currently supports only format='data'",
        ));
    }
    let source_path = parse_string(&source_path, "data.import sourcePath")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    copy_dir_recursive_on(
        authority.provider(),
        &authority.resolve(&source_path),
        &root,
    )
    .await?;
    let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
    Ok(dataset_object(&path, &manifest))
}

#[runtime_builtin(
    name = "data.export",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_export_builtin(
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
    let authority = DataPathAuthority::capture()?;
    copy_dir_recursive_on(
        authority.provider(),
        &authority.resolve(&path),
        &authority.resolve(&target_path),
    )
    .await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.list",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_list_builtin(path_prefix: Value) -> BuiltinResult<Value> {
    let prefix = parse_string(&path_prefix, "data.list prefix")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&prefix);
    let entries = authority.provider().read_dir(&root).await.map_err(|err| {
        data_error(format!(
            "data.list: failed to read '{}': {err}",
            root.display()
        ))
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
            values.push(Value::String(
                PathBuf::from(&prefix)
                    .join(entry.file_name())
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
    }
    let cols = values.len();
    make_cell(values, 1, cols).map_err(data_error)
}

#[runtime_builtin(
    name = "data.inspect",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_inspect_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.inspect path")?;
    let authority = DataPathAuthority::capture()?;
    let root = authority.resolve(&path);
    let manifest = crate::data::read_manifest_on(authority.provider(), &root).await?;
    let mut out = StructValue::new();
    out.fields.insert("path".to_string(), Value::String(path));
    out.fields
        .insert("id".to_string(), Value::String(manifest.dataset_id));
    out.fields.insert(
        "arrayCount".to_string(),
        Value::Num(manifest.arrays.len() as f64),
    );
    out.fields
        .insert("updatedAt".to_string(), Value::String(manifest.updated_at));
    Ok(Value::Struct(out))
}
