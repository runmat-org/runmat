//! Managed data receiver runtime bindings.

use super::*;
use crate::context::{host_effect::DatasetHandleOperation as Op, RuntimeContext};

#[runtime_builtin(
    name = "Dataset.path",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_path_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let owner = authority::resolve_object(runtime.as_ref(), &base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.path")?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetPath, &[], async {
        Ok(get_object_prop(obj, "__data_path")?.clone())
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.id",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_id_builtin(runtime: Option<RuntimeContext>, base: Value) -> BuiltinResult<Value> {
    let owner = authority::resolve_object(runtime.as_ref(), &base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.id")?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetId, &[], async {
        Ok(get_object_prop(obj, "__data_id")?.clone())
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.version",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_version_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let owner = authority::resolve_object(runtime.as_ref(), &base, HandleRole::Dataset)?;
    let obj = as_object(&base, "Dataset.version")?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetVersion, &[], async {
        Ok(get_object_prop(obj, "__data_version")?.clone())
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.arrays",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_arrays_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetArrays, &[], async {
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        let values: Vec<Value> = manifest
            .arrays
            .keys()
            .map(|k| Value::String(k.clone()))
            .collect();
        make_cell(values.clone(), 1, values.len()).map_err(data_error)
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.has_array",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_has_array_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    name: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let array_name = parse_string(&name, "Dataset.has_array name")?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetHasArray,
        &[&name],
        async {
            let manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            Ok(Value::Bool(manifest.arrays.contains_key(&array_name)))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.array",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_array_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    name: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let array_name = parse_string(&name, "Dataset.array name")?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetArray,
        &[&name],
        async {
            let manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            if !manifest.arrays.contains_key(&array_name) {
                return Err(data_error(format!(
                    "Dataset.array: array '{array_name}' not found"
                )));
            }
            let lease = authority::issue_child(&owner, HandleRole::Array(array_name.clone()))?;
            Ok(array_object(&owner.public_path, &array_name, lease))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.attrs",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_attrs_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetAttrs, &[], async {
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        Ok(attrs_to_struct(&manifest.attrs))
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.get_attr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_get_attr_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    key: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let key_text = parse_string(&key, "Dataset.get_attr key")?;
    let mut inputs = vec![&key];
    inputs.extend(rest.iter());
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetGetAttr,
        &inputs,
        async {
            let manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            if let Some(value) = manifest.attrs.get(&key_text) {
                return Ok(json_to_value(value));
            }
            Ok(rest.first().cloned().unwrap_or(Value::Num(0.0)))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.set_attr",
    binding_variant = "default",
    context = true,
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attr_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    key: Value,
    value: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let key_text = parse_string(&key, "Dataset.set_attr key")?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetSetAttr,
        &[&key, &value],
        async {
            let mut manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            manifest.attrs.insert(key_text, value_to_json(&value));
            manifest.updated_at = now_rfc3339();
            manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
            crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest)
                .await?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.set_attrs",
    binding_variant = "default",
    context = true,
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attrs_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    attrs: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let Value::Struct(incoming) = &attrs else {
        return Err(data_error("Dataset.set_attrs: attrs must be a struct"));
    };
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetSetAttrs,
        &[&attrs],
        async {
            let mut manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            for (k, v) in &incoming.fields {
                manifest.attrs.insert(k.clone(), value_to_json(v));
            }
            manifest.updated_at = now_rfc3339();
            manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
            crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest)
                .await?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.begin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_begin_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetBegin, &[], async {
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
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
    })
    .await
}

#[runtime_builtin(
    name = "Dataset.snapshot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_snapshot_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    label: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    let label_text = parse_string(&label, "Dataset.snapshot label")?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::DatasetSnapshot,
        &[&label],
        async {
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
            let dst = snapshots.join(format!("{}.manifest.json", sanitize_label(&label_text)));
            let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
            copy_file_on(owner.filesystem.as_ref(), &src, &dst, &mut budget).await?;
            Ok(Value::String(dst.to_string_lossy().to_string()))
        },
    )
    .await
}

#[runtime_builtin(
    name = "Dataset.refresh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_refresh_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let owner = dataset_owner(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::DatasetRefresh, &[], async {
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        let lease = authority::renew_dataset(&owner)?;
        let mut ds = dataset_object(&owner.public_path, &manifest, lease);
        hydrate_dataset_descriptor_on(owner.filesystem.as_ref(), &owner.root, &mut ds).await;
        Ok(ds)
    })
    .await
}
