//! Managed data receiver runtime bindings.

use super::*;

#[runtime_builtin(
    name = "Dataset.path",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_path_builtin(base: Value) -> BuiltinResult<Value> {
    authority::resolve_object(&base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.path")?;
    Ok(get_object_prop(obj, "__data_path")?.clone())
}

#[runtime_builtin(
    name = "Dataset.id",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_id_builtin(base: Value) -> BuiltinResult<Value> {
    authority::resolve_object(&base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.id")?;
    Ok(get_object_prop(obj, "__data_id")?.clone())
}

#[runtime_builtin(
    name = "Dataset.version",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_version_builtin(base: Value) -> BuiltinResult<Value> {
    authority::resolve_object(&base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.version")?;
    Ok(get_object_prop(obj, "__data_version")?.clone())
}

#[runtime_builtin(
    name = "Dataset.arrays",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_arrays_builtin(base: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let values: Vec<Value> = manifest
        .arrays
        .keys()
        .map(|k| Value::String(k.clone()))
        .collect();
    make_cell(values.clone(), 1, values.len()).map_err(data_error)
}

#[runtime_builtin(
    name = "Dataset.has_array",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_has_array_builtin(base: Value, name: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let name = parse_string(&name, "Dataset.has_array name")?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    Ok(Value::Bool(manifest.arrays.contains_key(&name)))
}

#[runtime_builtin(
    name = "Dataset.array",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_array_builtin(base: Value, name: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let name = parse_string(&name, "Dataset.array name")?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    if !manifest.arrays.contains_key(&name) {
        return Err(data_error(format!(
            "Dataset.array: array '{name}' not found"
        )));
    }
    let lease = authority::issue_child(&owner, HandleRole::Array(name.clone()))?;
    Ok(array_object(&owner.public_path, &name, lease))
}

#[runtime_builtin(
    name = "Dataset.attrs",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_attrs_builtin(base: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    Ok(attrs_to_struct(&manifest.attrs))
}

#[runtime_builtin(
    name = "Dataset.get_attr",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_get_attr_builtin(
    base: Value,
    key: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let key = parse_string(&key, "Dataset.get_attr key")?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    if let Some(value) = manifest.attrs.get(&key) {
        return Ok(json_to_value(value));
    }
    Ok(rest.first().cloned().unwrap_or(Value::Num(0.0)))
}

#[runtime_builtin(
    name = "Dataset.set_attr",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attr_builtin(base: Value, key: Value, value: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let key = parse_string(&key, "Dataset.set_attr key")?;
    let mut manifest =
        crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    manifest.attrs.insert(key, value_to_json(&value));
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "Dataset.set_attrs",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attrs_builtin(base: Value, attrs: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let Value::Struct(incoming) = attrs else {
        return Err(data_error("Dataset.set_attrs: attrs must be a struct"));
    };
    let mut manifest =
        crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    for (k, v) in incoming.fields {
        manifest.attrs.insert(k, value_to_json(&v));
    }
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "Dataset.begin",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_begin_builtin(base: Value, _rest: Vec<Value>) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let tx_id = new_tx_id()?;
    let lease = authority::issue_child(&owner, HandleRole::Transaction(tx_id.clone()))?;
    start_tx_with_id(
        tx_id.clone(),
        owner.public_path.clone(),
        manifest.txn_sequence,
    )?;
    tracing::info!(
        target: "runmat.data",
        dataset = owner.public_path,
        tx_id = tx_id,
        base_sequence = manifest.txn_sequence,
        "data transaction begin"
    );
    Ok(transaction_object(&owner.public_path, &tx_id, lease))
}

#[runtime_builtin(
    name = "Dataset.snapshot",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_snapshot_builtin(
    base: Value,
    label: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let label = parse_string(&label, "Dataset.snapshot label")?;
    let snapshots = owner.root.join(".snapshots");
    owner
        .filesystem
        .create_dir_all(&snapshots)
        .await
        .map_err(|source| {
            crate::data::data_error_with_source(
                format!("Dataset.snapshot: failed to create snapshots dir: {source}"),
                source,
            )
        })?;
    let src = manifest_path(&owner.root);
    let dst = snapshots.join(format!("{}.manifest.json", sanitize_label(&label)));
    let mut unmetered = None;
    copy_file_on(owner.filesystem.as_ref(), &src, &dst, &mut unmetered).await?;
    Ok(Value::String(dst.to_string_lossy().to_string()))
}

#[runtime_builtin(
    name = "Dataset.refresh",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_refresh_builtin(base: Value) -> BuiltinResult<Value> {
    let owner = dataset_owner(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let lease = authority::issue_dataset(
        owner.filesystem.clone(),
        owner.root.clone(),
        owner.public_path.clone(),
    )?;
    let mut ds = dataset_object(&owner.public_path, &manifest, lease);
    hydrate_dataset_descriptor_on(owner.filesystem.as_ref(), &owner.root, &mut ds).await;
    Ok(ds)
}
