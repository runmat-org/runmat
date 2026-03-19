//! Cloud-ready dataset persistence builtins (`data.*`).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

use runmat_builtins::{ObjectInstance, StructValue, Tensor, Value};
use runmat_filesystem::data_contract::{DataChunkDescriptor, DataChunkUploadRequest};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::data::{
    array_object, data_error, dataset_object, dataset_root, ensure_manifest_sequence,
    get_object_prop, manifest_path, manifest_version_token, now_rfc3339, parse_schema,
    parse_string, read_array_payload_async, read_manifest_async, remove_tx, sha256_hex, start_tx,
    transaction_object, with_tx, with_tx_mut, write_array_payload_async, write_manifest_async,
    DataArrayMeta, DataArrayPayload, DataChunkIndex, DataChunkIndexEntry, DataManifest,
    PendingCreateArray, PendingFill, PendingResize, PendingWrite, TxnStatus,
};
use crate::{make_cell, BuiltinResult};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::data")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "data.*",
    op_kind: GpuOpKind::Custom("io-data"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Dataset operations are host I/O and metadata orchestration.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::data")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "data.*",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Data builtins are side-effecting and not fusible.",
};

#[runtime_builtin(
    name = "data.create",
    category = "io/data",
    summary = "Create a typed dataset at a .data path.",
    keywords = "data,dataset,create,persistence",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_dataset_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_create_builtin(
    path: Value,
    schema: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.create path")?;
    let root = dataset_root(&path);
    let schema = parse_schema(&schema)?;
    let now = now_rfc3339();
    let mut arrays = BTreeMap::new();
    for (name, mut meta) in schema.arrays {
        let total = meta.shape.iter().copied().product::<usize>();
        let payload = DataArrayPayload {
            dtype: meta.dtype.clone(),
            shape: meta.shape.clone(),
            values: vec![0.0; total],
        };
        let (payload_path, chunk_index_path) =
            write_array_payload_async(&root, &name, &payload, &meta.chunk_shape).await?;
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
    write_manifest_async(&root, &manifest).await?;
    Ok(dataset_object(&path, &manifest))
}

#[runtime_builtin(
    name = "data.open",
    category = "io/data",
    summary = "Open a dataset handle from a .data path.",
    keywords = "data,dataset,open,persistence",
    type_resolver(crate::builtins::io::type_resolvers::data_dataset_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_open_builtin(path: Value, _rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.open path")?;
    let root = dataset_root(&path);
    let manifest = read_manifest_async(&root).await?;
    let mut ds = dataset_object(&path, &manifest);
    hydrate_dataset_descriptor_async(&path, &mut ds).await;
    Ok(ds)
}

#[runtime_builtin(
    name = "data.exists",
    category = "io/data",
    summary = "Check if dataset exists.",
    keywords = "data,dataset,exists",
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_exists_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.exists path")?;
    let root = dataset_root(&path);
    let exists = runmat_filesystem::metadata_async(manifest_path(&root))
        .await
        .is_ok();
    Ok(Value::Bool(exists))
}

#[runtime_builtin(
    name = "data.delete",
    category = "io/data",
    summary = "Delete a dataset path.",
    keywords = "data,dataset,delete",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_delete_builtin(path: Value, _rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.delete path")?;
    let root = dataset_root(&path);
    runmat_filesystem::remove_dir_all_async(&root)
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
    category = "io/data",
    summary = "Copy dataset to new path.",
    keywords = "data,dataset,copy",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_copy_builtin(
    from_path: Value,
    to_path: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.copy fromPath")?;
    let to = parse_string(&to_path, "data.copy toPath")?;
    copy_dir_recursive(&dataset_root(&from), &dataset_root(&to)).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.move",
    category = "io/data",
    summary = "Move dataset to new path.",
    keywords = "data,dataset,move",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_move_builtin(
    from_path: Value,
    to_path: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let from = parse_string(&from_path, "data.move fromPath")?;
    let to = parse_string(&to_path, "data.move toPath")?;
    runmat_filesystem::rename_async(dataset_root(&from), dataset_root(&to))
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
    category = "io/data",
    summary = "Import an existing dataset file path.",
    keywords = "data,dataset,import",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_dataset_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_import_builtin(
    path: Value,
    format: Value,
    source_path: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.import path")?;
    let format = parse_string(&format, "data.import format")?;
    if !format.eq_ignore_ascii_case("data") {
        return Err(data_error(
            "data.import currently supports only format='data'",
        ));
    }
    let source_path = parse_string(&source_path, "data.import sourcePath")?;
    copy_dir_recursive(&dataset_root(&source_path), &dataset_root(&path)).await?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    Ok(dataset_object(&path, &manifest))
}

#[runtime_builtin(
    name = "data.export",
    category = "io/data",
    summary = "Export dataset to target path.",
    keywords = "data,dataset,export",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_export_builtin(
    path: Value,
    format: Value,
    target_path: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.export path")?;
    let format = parse_string(&format, "data.export format")?;
    if !format.eq_ignore_ascii_case("data") {
        return Err(data_error(
            "data.export currently supports only format='data'",
        ));
    }
    let target_path = parse_string(&target_path, "data.export targetPath")?;
    copy_dir_recursive(&dataset_root(&path), &dataset_root(&target_path)).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "data.list",
    category = "io/data",
    summary = "List dataset paths under a prefix.",
    keywords = "data,dataset,list",
    type_resolver(crate::builtins::io::type_resolvers::data_cell_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_list_builtin(path_prefix: Value, _rest: Vec<Value>) -> BuiltinResult<Value> {
    let prefix = parse_string(&path_prefix, "data.list prefix")?;
    let root = PathBuf::from(prefix);
    let entries = runmat_filesystem::read_dir_async(&root)
        .await
        .map_err(|err| {
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
        if runmat_filesystem::metadata_async(candidate.join("manifest.json"))
            .await
            .is_ok()
        {
            values.push(Value::String(candidate.to_string_lossy().to_string()));
        }
    }
    let cols = values.len();
    make_cell(values, 1, cols).map_err(data_error)
}

#[runtime_builtin(
    name = "data.inspect",
    category = "io/data",
    summary = "Inspect dataset metadata and schema fields.",
    keywords = "data,dataset,inspect,schema",
    type_resolver(crate::builtins::io::type_resolvers::data_struct_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_inspect_builtin(path: Value) -> BuiltinResult<Value> {
    let path = parse_string(&path, "data.inspect path")?;
    let root = dataset_root(&path);
    let manifest = read_manifest_async(&root).await?;
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

#[runtime_builtin(
    name = "Dataset.path",
    category = "io/data",
    summary = "Return dataset path.",
    keywords = "dataset,path",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_path_builtin(base: Value) -> BuiltinResult<Value> {
    let obj = as_object(&base, "Dataset.path")?;
    Ok(get_object_prop(obj, "__data_path")?.clone())
}

#[runtime_builtin(
    name = "Dataset.id",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_id_builtin(base: Value) -> BuiltinResult<Value> {
    let obj = as_object(&base, "Dataset.id")?;
    Ok(get_object_prop(obj, "__data_id")?.clone())
}

#[runtime_builtin(
    name = "Dataset.version",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_version_builtin(base: Value) -> BuiltinResult<Value> {
    let obj = as_object(&base, "Dataset.version")?;
    Ok(get_object_prop(obj, "__data_version")?.clone())
}

#[runtime_builtin(
    name = "Dataset.arrays",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_cell_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_arrays_builtin(base: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.arrays")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let values: Vec<Value> = manifest
        .arrays
        .keys()
        .map(|k| Value::String(k.clone()))
        .collect();
    make_cell(values.clone(), 1, values.len()).map_err(data_error)
}

#[runtime_builtin(
    name = "Dataset.has_array",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_has_array_builtin(base: Value, name: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.has_array")?;
    let name = parse_string(&name, "Dataset.has_array name")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    Ok(Value::Bool(manifest.arrays.contains_key(&name)))
}

#[runtime_builtin(
    name = "Dataset.array",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_array_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_array_builtin(base: Value, name: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.array")?;
    let name = parse_string(&name, "Dataset.array name")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    if !manifest.arrays.contains_key(&name) {
        return Err(data_error(format!(
            "Dataset.array: array '{name}' not found"
        )));
    }
    Ok(array_object(&path, &name))
}

#[runtime_builtin(
    name = "Dataset.attrs",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_struct_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_attrs_builtin(base: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.attrs")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    Ok(attrs_to_struct(&manifest.attrs))
}

#[runtime_builtin(
    name = "Dataset.get_attr",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_unknown_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_get_attr_builtin(
    base: Value,
    key: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.get_attr")?;
    let key = parse_string(&key, "Dataset.get_attr key")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    if let Some(value) = manifest.attrs.get(&key) {
        return Ok(json_to_value(value));
    }
    Ok(rest.first().cloned().unwrap_or(Value::Num(0.0)))
}

#[runtime_builtin(
    name = "Dataset.set_attr",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attr_builtin(base: Value, key: Value, value: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.set_attr")?;
    let key = parse_string(&key, "Dataset.set_attr key")?;
    let root = dataset_root(&path);
    let mut manifest = read_manifest_async(&root).await?;
    manifest.attrs.insert(key, value_to_json(&value));
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "Dataset.set_attrs",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_set_attrs_builtin(base: Value, attrs: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.set_attrs")?;
    let Value::Struct(incoming) = attrs else {
        return Err(data_error("Dataset.set_attrs: attrs must be a struct"));
    };
    let root = dataset_root(&path);
    let mut manifest = read_manifest_async(&root).await?;
    for (k, v) in incoming.fields {
        manifest.attrs.insert(k, value_to_json(&v));
    }
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "Dataset.begin",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_tx_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_begin_builtin(base: Value, _rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.begin")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let tx_id = start_tx(path.clone(), manifest.txn_sequence);
    tracing::info!(
        target: "runmat.data",
        dataset = path,
        tx_id = tx_id,
        base_sequence = manifest.txn_sequence,
        "data transaction begin"
    );
    Ok(transaction_object(&path, &tx_id))
}

#[runtime_builtin(
    name = "Dataset.snapshot",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_snapshot_builtin(
    base: Value,
    label: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.snapshot")?;
    let label = parse_string(&label, "Dataset.snapshot label")?;
    let root = dataset_root(&path);
    let snapshots = root.join(".snapshots");
    runmat_filesystem::create_dir_all_async(&snapshots)
        .await
        .map_err(|err| {
            data_error(format!(
                "Dataset.snapshot: failed to create snapshots dir: {err}"
            ))
        })?;
    let src = manifest_path(&root);
    let dst = snapshots.join(format!("{}.manifest.json", sanitize_label(&label)));
    copy_file(&src, &dst).await?;
    Ok(Value::String(dst.to_string_lossy().to_string()))
}

#[runtime_builtin(
    name = "Dataset.refresh",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_dataset_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn dataset_refresh_builtin(base: Value) -> BuiltinResult<Value> {
    let path = dataset_path_from_object(&base, "Dataset.refresh")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let mut ds = dataset_object(&path, &manifest);
    hydrate_dataset_descriptor_async(&path, &mut ds).await;
    Ok(ds)
}

#[runtime_builtin(
    name = "DataArray.name",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_name_builtin(base: Value) -> BuiltinResult<Value> {
    let obj = as_object(&base, "DataArray.name")?;
    Ok(get_object_prop(obj, "__array_name")?.clone())
}

#[runtime_builtin(
    name = "DataArray.dtype",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_dtype_builtin(base: Value) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.dtype")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.dtype: array '{name}' not found")))?;
    Ok(Value::String(meta.dtype.clone()))
}

#[runtime_builtin(
    name = "DataArray.shape",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_shape_tensor_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_shape_builtin(base: Value) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.shape")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.shape: array '{name}' not found")))?;
    let values = meta.shape.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let tensor = Tensor::new(values, vec![1, meta.shape.len()])
        .map_err(|err| data_error(format!("DataArray.shape: {err}")))?;
    Ok(Value::Tensor(tensor))
}

#[runtime_builtin(
    name = "DataArray.rank",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_int_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_rank_builtin(base: Value) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.rank")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.rank: array '{name}' not found")))?;
    Ok(Value::Num(meta.shape.len() as f64))
}

#[runtime_builtin(
    name = "DataArray.chunk_shape",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_shape_tensor_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_chunk_shape_builtin(base: Value) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.chunk_shape")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.chunk_shape: array '{name}' not found")))?;
    let values = meta
        .chunk_shape
        .iter()
        .map(|v| *v as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(values, vec![1, meta.chunk_shape.len()])
        .map_err(|err| data_error(format!("DataArray.chunk_shape: {err}")))?;
    Ok(Value::Tensor(tensor))
}

#[runtime_builtin(
    name = "DataArray.codec",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_codec_builtin(base: Value) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.codec")?;
    let manifest = read_manifest_async(&dataset_root(&path)).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.codec: array '{name}' not found")))?;
    Ok(Value::String(meta.codec.clone()))
}

#[runtime_builtin(
    name = "DataArray.read",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_tensor_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_read_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.read")?;
    let root = dataset_root(&path);
    let manifest = read_manifest_async(&root).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.read: array '{name}' not found")))?;
    let payload = read_array_payload_async(&root, meta).await?;
    let sliced = if let Some(slice_spec) = rest.first() {
        read_slice_payload(&payload, slice_spec)?
    } else {
        payload
    };
    let tensor = Tensor::new(sliced.values, sliced.shape)
        .map_err(|err| data_error(format!("DataArray.read: {err}")))?;
    Ok(Value::Tensor(tensor))
}

#[runtime_builtin(
    name = "DataArray.write",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_write_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.write")?;
    let (slice_spec, value) = match rest.as_slice() {
        [v] => (None, v),
        [slice, v] => (Some(slice), v),
        _ => {
            return Err(data_error(
                "DataArray.write expects values or (sliceSpec, values) arguments",
            ))
        }
    };
    write_array_full_async(&path, &name, slice_spec, value).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataArray.resize",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_resize_builtin(
    base: Value,
    new_shape: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.resize")?;
    let shape = parse_shape_from_value(&new_shape)?;
    let root = dataset_root(&path);
    let mut manifest = read_manifest_async(&root).await?;
    let meta = manifest
        .arrays
        .get_mut(&name)
        .ok_or_else(|| data_error(format!("DataArray.resize: array '{name}' not found")))?;
    meta.shape = shape.clone();
    let payload = DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: shape.clone(),
        values: vec![0.0; shape.iter().copied().product()],
    };
    let (payload_path, chunk_index_path) =
        write_array_payload_async(&root, &name, &payload, &meta.chunk_shape).await?;
    meta.data_path = make_rel_data_path(&root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(&root, &chunk_index_path)?);
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataArray.fill",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_fill_builtin(
    base: Value,
    value: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let (path, name) = array_identity(&base, "DataArray.fill")?;
    let root = dataset_root(&path);
    let mut manifest = read_manifest_async(&root).await?;
    let meta = manifest
        .arrays
        .get_mut(&name)
        .ok_or_else(|| data_error(format!("DataArray.fill: array '{name}' not found")))?;
    let scalar = scalar_to_f64(&value)?;
    let payload = DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: meta.shape.clone(),
        values: vec![scalar; meta.shape.iter().copied().product()],
    };
    let (payload_path, chunk_index_path) =
        write_array_payload_async(&root, &name, &payload, &meta.chunk_shape).await?;
    meta.data_path = make_rel_data_path(&root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(&root, &chunk_index_path)?);
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.id",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_id_builtin(base: Value) -> BuiltinResult<Value> {
    let obj = as_object(&base, "DataTransaction.id")?;
    Ok(get_object_prop(obj, "__tx_id")?.clone())
}

#[runtime_builtin(
    name = "DataTransaction.write",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_write_builtin(
    base: Value,
    array_name: Value,
    slice: Value,
    values: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.write")?;
    let array_name = parse_string(&array_name, "DataTransaction.write arrayName")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error("DataTransaction.write: transaction is not open"));
        }
        tx.writes.push(PendingWrite {
            array: array_name,
            slice_spec: Some(slice),
            value: values,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.set_attr",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_set_attr_builtin(base: Value, key: Value, value: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.set_attr")?;
    let key = parse_string(&key, "DataTransaction.set_attr key")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.set_attr: transaction is not open",
            ));
        }
        tx.attrs.insert(key, value);
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.set_attrs",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_set_attrs_builtin(base: Value, attrs: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.set_attrs")?;
    let Value::Struct(incoming) = attrs else {
        return Err(data_error(
            "DataTransaction.set_attrs: attrs must be struct",
        ));
    };
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.set_attrs: transaction is not open",
            ));
        }
        for (k, v) in incoming.fields {
            tx.attrs.insert(k, v);
        }
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.resize",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_resize_builtin(
    base: Value,
    array_name: Value,
    new_shape: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.resize")?;
    let array_name = parse_string(&array_name, "DataTransaction.resize arrayName")?;
    let shape = parse_shape_from_value(&new_shape)?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.resize: transaction is not open",
            ));
        }
        tx.resizes.push(PendingResize {
            array: array_name,
            shape,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.fill",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_fill_builtin(
    base: Value,
    array_name: Value,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.fill")?;
    let array_name = parse_string(&array_name, "DataTransaction.fill arrayName")?;
    let slice_spec = rest.first().cloned();
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error("DataTransaction.fill: transaction is not open"));
        }
        tx.fills.push(PendingFill {
            array: array_name,
            slice_spec,
            value,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.delete_array",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_delete_array_builtin(base: Value, array_name: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.delete_array")?;
    let array_name = parse_string(&array_name, "DataTransaction.delete_array arrayName")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.delete_array: transaction is not open",
            ));
        }
        tx.delete_arrays.push(array_name);
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.create_array",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_create_array_builtin(
    base: Value,
    array_name: Value,
    meta: Value,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.create_array")?;
    let array_name = parse_string(&array_name, "DataTransaction.create_array arrayName")?;
    let meta = parse_array_meta(&array_name, &meta)?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.create_array: transaction is not open",
            ));
        }
        tx.create_arrays.push(PendingCreateArray {
            array: array_name,
            meta,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.commit",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_commit_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.commit")?;
    let (dataset_path, base_sequence, writes, resizes, fills, create_arrays, delete_arrays, attrs) =
        with_tx(&tx_id, |tx| {
            if tx.status != TxnStatus::Open {
                return Err(data_error(
                    "DataTransaction.commit: transaction is not open",
                ));
            }
            Ok((
                tx.dataset_path.clone(),
                tx.base_sequence,
                tx.writes.clone(),
                tx.resizes.clone(),
                tx.fills.clone(),
                tx.create_arrays.clone(),
                tx.delete_arrays.clone(),
                tx.attrs.clone(),
            ))
        })?;

    let write_ops = writes.len();
    let resize_ops = resizes.len();
    let fill_ops = fills.len();
    let create_ops = create_arrays.len();
    let delete_ops = delete_arrays.len();
    let attr_updates = attrs.len();

    let root = dataset_root(&dataset_path);
    let mut manifest = read_manifest_async(&root).await?;
    ensure_manifest_sequence(base_sequence, &manifest)?;
    if let Some(Value::Struct(options)) = rest.first() {
        if let Some(expected) = options.fields.get("if_manifest") {
            let expected = parse_string(expected, "DataTransaction.commit if_manifest")?;
            let actual = manifest_version_token(&manifest);
            if expected != actual {
                tracing::warn!(
                    target: "runmat.data",
                    tx_id = tx_id,
                    expected_manifest = expected,
                    actual_manifest = actual,
                    "data transaction manifest conflict"
                );
                return Err(data_error(
                    "MANIFEST_CONFLICT: if_manifest precondition failed",
                ));
            }
        }
    }
    for create in create_arrays {
        create_array_in_manifest(&root, &mut manifest, &create.array, create.meta).await?;
    }
    for resize in resizes {
        resize_array_in_manifest(&root, &mut manifest, &resize.array, resize.shape).await?;
    }
    for fill in fills {
        fill_array_in_manifest(
            &root,
            &mut manifest,
            &fill.array,
            fill.slice_spec.as_ref(),
            &fill.value,
        )
        .await?;
    }
    for write in writes {
        apply_write_to_manifest_async(
            &root,
            &mut manifest,
            &write.array,
            write.slice_spec.as_ref(),
            &write.value,
        )
        .await?;
    }
    for array_name in delete_arrays {
        delete_array_in_manifest_async(&root, &mut manifest, &array_name).await?;
    }
    for (k, v) in attrs {
        manifest.attrs.insert(k, value_to_json(&v));
    }
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await?;
    with_tx_mut(&tx_id, |tx| {
        tx.status = TxnStatus::Committed;
        Ok(())
    })?;
    tracing::info!(
        target: "runmat.data",
        dataset = dataset_path,
        tx_id = tx_id,
        write_ops = write_ops,
        resize_ops = resize_ops,
        fill_ops = fill_ops,
        create_ops = create_ops,
        delete_ops = delete_ops,
        attr_updates = attr_updates,
        next_sequence = manifest.txn_sequence,
        "data transaction commit"
    );
    remove_tx(&tx_id);
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "commit",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_commit_alias_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match &base {
        Value::Object(obj) if obj.class_name == "DataTransaction" => {
            data_tx_commit_builtin(base, rest).await
        }
        Value::HandleObject(handle) if handle.class_name == "DataTransaction" => {
            data_tx_commit_builtin(base, rest).await
        }
        _ => Err(data_error(
            "commit: receiver must be a DataTransaction (use tx = ds.begin())",
        )),
    }
}

#[runtime_builtin(
    name = "DataTransaction.abort",
    category = "io/data",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::data_bool_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_abort_builtin(base: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.abort")?;
    with_tx_mut(&tx_id, |tx| {
        tx.status = TxnStatus::Aborted;
        Ok(())
    })?;
    tracing::info!(
        target: "runmat.data",
        tx_id = tx_id,
        "data transaction abort"
    );
    remove_tx(&tx_id);
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.status",
    category = "io/data",
    type_resolver(crate::builtins::io::type_resolvers::data_string_type),
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_status_builtin(base: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.status")?;
    with_tx(&tx_id, |tx| {
        let status = match tx.status {
            TxnStatus::Open => "open",
            TxnStatus::Committed => "committed",
            TxnStatus::Aborted => "aborted",
        };
        Ok(Value::String(status.to_string()))
    })
}

fn dataset_path_from_object(base: &Value, context: &str) -> BuiltinResult<String> {
    let obj = as_object(base, context)?;
    parse_string(get_object_prop(obj, "__data_path")?, context)
}

fn tx_id_from_object(base: &Value, context: &str) -> BuiltinResult<String> {
    let obj = as_object(base, context)?;
    parse_string(get_object_prop(obj, "__tx_id")?, context)
}

fn array_identity(base: &Value, context: &str) -> BuiltinResult<(String, String)> {
    let obj = as_object(base, context)?;
    let path = parse_string(get_object_prop(obj, "__data_path")?, context)?;
    let name = parse_string(get_object_prop(obj, "__array_name")?, context)?;
    Ok((path, name))
}

fn as_object<'a>(value: &'a Value, context: &str) -> BuiltinResult<&'a ObjectInstance> {
    match value {
        Value::Object(obj) => Ok(obj),
        _ => Err(data_error(format!("{context}: expected object receiver"))),
    }
}

async fn hydrate_dataset_descriptor_async(path: &str, dataset: &mut Value) {
    let request = runmat_filesystem::data_contract::DataManifestRequest {
        path: path.to_string(),
        version: None,
    };
    let descriptor = match runmat_filesystem::data_manifest_descriptor_async(&request).await {
        Ok(descriptor) => descriptor,
        Err(_) => return,
    };
    let Value::Object(obj) = dataset else {
        return;
    };
    if !descriptor.dataset_id.is_empty() {
        obj.properties.insert(
            "__data_id".to_string(),
            Value::String(descriptor.dataset_id),
        );
    }
    obj.properties.insert(
        "__data_version".to_string(),
        Value::String(format!(
            "{}:{}",
            descriptor.updated_at, descriptor.txn_sequence
        )),
    );
}

fn sanitize_label(label: &str) -> String {
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

async fn copy_file(src: &PathBuf, dst: &PathBuf) -> BuiltinResult<()> {
    let bytes = runmat_filesystem::read_async(src)
        .await
        .map_err(|err| data_error(format!("failed to open '{}': {err}", src.display())))?;
    let parent = dst.parent().ok_or_else(|| {
        data_error(format!(
            "invalid destination path '{}': missing parent",
            dst.display()
        ))
    })?;
    runmat_filesystem::create_dir_all_async(parent)
        .await
        .map_err(|err| data_error(format!("failed to create '{}': {err}", parent.display())))?;
    runmat_filesystem::write_async(dst, &bytes)
        .await
        .map_err(|err| {
            data_error(format!(
                "failed to copy '{}' -> '{}': {err}",
                src.display(),
                dst.display()
            ))
        })?;
    Ok(())
}

fn make_rel_data_path(
    root: &std::path::Path,
    payload_path: &std::path::Path,
) -> BuiltinResult<String> {
    let rel = payload_path
        .strip_prefix(root)
        .map_err(|err| data_error(format!("failed to compute relative data path: {err}")))?;
    Ok(rel.to_string_lossy().to_string())
}

async fn create_array_in_manifest(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    mut meta: DataArrayMeta,
) -> BuiltinResult<()> {
    if manifest.arrays.contains_key(array_name) {
        return Err(data_error(format!(
            "DataTransaction.create_array: array '{array_name}' already exists"
        )));
    }
    let payload = DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: meta.shape.clone(),
        values: vec![0.0; meta.shape.iter().copied().product()],
    };
    let (payload_path, chunk_index_path) =
        write_array_payload_async(root, array_name, &payload, &meta.chunk_shape).await?;
    meta.data_path = make_rel_data_path(root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    manifest.arrays.insert(array_name.to_string(), meta);
    Ok(())
}

async fn resize_array_in_manifest(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    shape: Vec<usize>,
) -> BuiltinResult<()> {
    let meta = manifest
        .arrays
        .get_mut(array_name)
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;
    meta.shape = shape.clone();
    let payload = DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: shape.clone(),
        values: vec![0.0; shape.iter().copied().product()],
    };
    let (payload_path, chunk_index_path) =
        write_array_payload_async(root, array_name, &payload, &meta.chunk_shape).await?;
    meta.data_path = make_rel_data_path(root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    Ok(())
}

async fn fill_array_in_manifest(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
) -> BuiltinResult<()> {
    let meta: DataArrayMeta = manifest
        .arrays
        .get(array_name)
        .cloned()
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;
    let scalar = scalar_to_f64(value)?;
    let payload = read_array_payload_async(root, &meta).await?;
    let next_payload = if let Some(slice_spec) = slice_spec {
        let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
        let target_shape: Vec<usize> = ranges
            .iter()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        let rhs = Value::Tensor(
            Tensor::new(
                vec![scalar; target_shape.iter().copied().product()],
                target_shape,
            )
            .map_err(|err| data_error(format!("DataTransaction.fill: {err}")))?,
        );
        write_slice_payload(&payload, slice_spec, &rhs)?
    } else {
        DataArrayPayload {
            dtype: payload.dtype,
            shape: payload.shape.clone(),
            values: vec![scalar; payload.shape.iter().copied().product()],
        }
    };
    let (payload_path, chunk_index_path) =
        write_array_payload_async(root, array_name, &next_payload, &meta.chunk_shape).await?;
    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = next_payload.shape.clone();
        updated.data_path = make_rel_data_path(root, &payload_path)?;
        updated.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    }
    Ok(())
}

async fn delete_array_in_manifest_async(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
) -> BuiltinResult<()> {
    let removed = manifest.arrays.remove(array_name);
    if removed.is_none() {
        return Err(data_error(format!(
            "DataTransaction.delete_array: array '{array_name}' not found"
        )));
    }
    let array_dir = root.join("arrays").join(array_name);
    if runmat_filesystem::metadata_async(&array_dir).await.is_ok() {
        runmat_filesystem::remove_dir_all_async(&array_dir)
            .await
            .map_err(|err| {
                data_error(format!(
                    "DataTransaction.delete_array: failed to remove '{}': {err}",
                    array_dir.display()
                ))
            })?;
    }
    Ok(())
}

fn parse_array_meta(array_name: &str, meta: &Value) -> BuiltinResult<DataArrayMeta> {
    let Value::Struct(meta_struct) = meta else {
        return Err(data_error(
            "DataTransaction.create_array: meta must be a struct",
        ));
    };
    let dtype = meta_struct
        .fields
        .get("dtype")
        .map(|v| parse_string(v, "DataTransaction.create_array dtype"))
        .transpose()?
        .unwrap_or_else(|| "f64".to_string());
    let shape = meta_struct
        .fields
        .get("shape")
        .map(parse_shape_from_value)
        .transpose()?
        .unwrap_or_else(|| vec![0, 0]);
    let chunk_shape = meta_struct
        .fields
        .get("chunk")
        .map(parse_shape_from_value)
        .transpose()?
        .unwrap_or_else(|| default_chunk_shape(&shape));
    let codec = meta_struct
        .fields
        .get("codec")
        .map(|v| parse_string(v, "DataTransaction.create_array codec"))
        .transpose()?
        .unwrap_or_else(|| "zstd".to_string());
    Ok(DataArrayMeta {
        dtype,
        shape,
        chunk_shape,
        order: "column_major".to_string(),
        codec,
        chunk_index_path: Some(format!("arrays/{array_name}/chunks/index.json")),
        data_path: format!("arrays/{array_name}/data.f64.json"),
    })
}

fn default_chunk_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1024];
    }
    let mut out = shape.to_vec();
    if out.len() == 1 {
        out[0] = out[0].clamp(1, 65_536);
        return out;
    }
    out[0] = out[0].clamp(1, 256);
    out[1] = out[1].clamp(1, 256);
    for dim in out.iter_mut().skip(2) {
        *dim = (*dim).clamp(1, 8);
    }
    out
}

#[async_recursion::async_recursion(?Send)]
async fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> BuiltinResult<()> {
    let metadata = runmat_filesystem::metadata_async(src)
        .await
        .map_err(|err| data_error(format!("failed to stat '{}': {err}", src.display())))?;
    if !metadata.is_dir() {
        return Err(data_error(format!(
            "expected dataset directory at '{}'",
            src.display()
        )));
    }
    runmat_filesystem::create_dir_all_async(dst)
        .await
        .map_err(|err| data_error(format!("failed to create '{}': {err}", dst.display())))?;
    for entry in runmat_filesystem::read_dir_async(src)
        .await
        .map_err(|err| data_error(format!("failed to read '{}': {err}", src.display())))?
    {
        let entry_src = entry.path().to_path_buf();
        let entry_dst = dst.join(entry.file_name());
        if entry.is_dir() {
            copy_dir_recursive(&entry_src, &entry_dst).await?;
            continue;
        }
        copy_file(&entry_src, &entry_dst).await?;
    }
    Ok(())
}

fn parse_shape_from_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => {
            let mut out = Vec::with_capacity(t.data.len());
            for v in &t.data {
                if !v.is_finite() || *v < 0.0 {
                    return Err(data_error(
                        "shape dimensions must be non-negative finite numbers",
                    ));
                }
                out.push(*v as usize);
            }
            Ok(out)
        }
        Value::Num(v) => {
            if !v.is_finite() || *v < 0.0 {
                return Err(data_error(
                    "shape dimensions must be non-negative finite numbers",
                ));
            }
            Ok(vec![*v as usize])
        }
        Value::Int(v) => {
            let n = v.to_i64();
            if n < 0 {
                return Err(data_error("shape dimensions must be non-negative"));
            }
            Ok(vec![n as usize])
        }
        _ => Err(data_error("shape must be a numeric vector")),
    }
}

fn scalar_to_f64(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(v) => Ok(v.to_i64() as f64),
        _ => Err(data_error("expected numeric scalar")),
    }
}

async fn write_array_full_async(
    dataset_path: &str,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
) -> BuiltinResult<()> {
    let root = dataset_root(dataset_path);
    let mut manifest = read_manifest_async(&root).await?;
    apply_write_to_manifest_async(&root, &mut manifest, array_name, slice_spec, value).await?;
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    write_manifest_async(&root, &manifest).await
}

async fn apply_write_to_manifest_async(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
) -> BuiltinResult<()> {
    let meta: DataArrayMeta = manifest
        .arrays
        .get(array_name)
        .cloned()
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;

    if let Some(slice_spec) = slice_spec {
        if apply_slice_write_chunked_async(root, manifest, array_name, &meta, slice_spec, value)
            .await?
        {
            return Ok(());
        }
    }

    let payload = read_array_payload_async(root, &meta).await?;
    let mut next_payload = payload.clone();
    if let Some(slice_spec) = slice_spec {
        next_payload = write_slice_payload(&payload, slice_spec, value)?;
    } else {
        let (shape, values) = value_to_tensor_shape_values(value)?;
        next_payload.shape = shape;
        next_payload.values = values;
    }

    let (payload_path, chunk_index_path) =
        write_array_payload_async(root, array_name, &next_payload, &meta.chunk_shape).await?;
    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = next_payload.shape.clone();
        updated.data_path = make_rel_data_path(root, &payload_path)?;
        updated.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    }
    Ok(())
}

async fn apply_slice_write_chunked_async(
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    meta: &DataArrayMeta,
    slice_spec: &Value,
    value: &Value,
) -> BuiltinResult<bool> {
    let Some(index_rel_path) = &meta.chunk_index_path else {
        return Ok(false);
    };
    let index_path = root.join(index_rel_path);
    if runmat_filesystem::metadata_async(&index_path)
        .await
        .is_err()
    {
        return Ok(false);
    }
    let ranges = parse_slice_spec(slice_spec, &meta.shape)?;
    let rhs_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let (actual_rhs_shape, rhs_values) = value_to_tensor_shape_values(value)?;
    if actual_rhs_shape != rhs_shape {
        return Err(data_error(format!(
            "SHAPE_MISMATCH: rhs shape {:?} must match target slice shape {:?}",
            actual_rhs_shape, rhs_shape
        )));
    }

    let index_bytes = runmat_filesystem::read_async(&index_path)
        .await
        .map_err(|err| {
            data_error(format!(
                "failed to read chunk index '{}': {err}",
                index_path.display()
            ))
        })?;
    let mut chunk_index: DataChunkIndex = serde_json::from_slice(&index_bytes).map_err(|err| {
        data_error(format!(
            "failed to parse chunk index '{}': {err}",
            index_path.display()
        ))
    })?;

    let mut pos_by_key = HashMap::new();
    for (idx, entry) in chunk_index.chunks.iter().enumerate() {
        pos_by_key.insert(entry.key.clone(), idx);
    }
    let touched = touched_chunk_coords(&ranges, &meta.chunk_shape, &meta.shape);
    let mut upload_batch = Vec::<(DataChunkDescriptor, Vec<u8>)>::new();

    for coords in touched {
        let key = chunk_key(&coords);
        let chunk_start = chunk_start_for_coords(&coords, &meta.chunk_shape);
        let chunk_extent = chunk_extent_for_start(&chunk_start, &meta.chunk_shape, &meta.shape);
        let intersection = chunk_intersection(&ranges, &chunk_start, &chunk_extent);
        if intersection.is_empty() {
            continue;
        }

        let (entry_index, existed, mut entry, mut chunk_payload) = load_or_init_chunk(
            root,
            array_name,
            &key,
            &coords,
            &chunk_extent,
            &pos_by_key,
            &chunk_index,
        )
        .await?;

        let mut local = vec![0usize; intersection.len()];
        let intersection_shape: Vec<usize> = intersection
            .iter()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        loop {
            let mut global = Vec::with_capacity(intersection.len());
            for dim in 0..intersection.len() {
                global.push(intersection[dim].start + local[dim]);
            }
            let rhs_index: Vec<usize> = global
                .iter()
                .enumerate()
                .map(|(dim, g)| g.saturating_sub(ranges[dim].start))
                .collect();
            let chunk_index_local: Vec<usize> = global
                .iter()
                .enumerate()
                .map(|(dim, g)| g.saturating_sub(chunk_start[dim]))
                .collect();
            let rhs_linear = linear_index_column_major(&rhs_index, &rhs_shape)?;
            let chunk_linear = linear_index_column_major(&chunk_index_local, &chunk_extent)?;
            chunk_payload.values[chunk_linear] = rhs_values[rhs_linear];
            if !advance_index(&mut local, &intersection_shape) {
                break;
            }
        }

        let chunk_bytes = serde_json::to_vec(&chunk_payload)
            .map_err(|err| data_error(format!("failed to encode chunk payload: {err}")))?;
        let chunk_path = root.join(&entry.data_path);
        runmat_filesystem::write_async(&chunk_path, &chunk_bytes)
            .await
            .map_err(|err| {
                data_error(format!(
                    "failed to write chunk payload '{}': {err}",
                    chunk_path.display()
                ))
            })?;

        entry.coords = coords.clone();
        entry.shape = chunk_extent.clone();
        entry.bytes_raw = chunk_bytes.len() as u64;
        entry.bytes_stored = chunk_bytes.len() as u64;
        entry.hash = sha256_hex(&chunk_bytes);
        if existed {
            chunk_index.chunks[entry_index] = entry.clone();
        } else {
            chunk_index.chunks.push(entry.clone());
            pos_by_key.insert(key.clone(), chunk_index.chunks.len() - 1);
        }
        upload_batch.push((
            DataChunkDescriptor {
                key: key.clone(),
                object_id: entry.object_id.clone(),
                hash: entry.hash.clone(),
                bytes_raw: entry.bytes_raw,
                bytes_stored: entry.bytes_stored,
            },
            chunk_bytes,
        ));
    }

    maybe_upload_chunk_batch_async(root, array_name, upload_batch).await?;
    tracing::info!(
        target: "runmat.data",
        dataset = %root.display(),
        array = array_name,
        touched_chunks = chunk_index.chunks.len(),
        "chunked slice write committed"
    );
    let index_write = serde_json::to_vec(&chunk_index)
        .map_err(|err| data_error(format!("failed to encode chunk index json: {err}")))?;
    runmat_filesystem::write_async(&index_path, &index_write)
        .await
        .map_err(|err| {
            data_error(format!(
                "failed to write chunk index '{}': {err}",
                index_path.display()
            ))
        })?;

    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = meta.shape.clone();
        updated.chunk_index_path = Some(index_rel_path.clone());
    }
    Ok(true)
}

fn value_to_tensor_shape_values(value: &Value) -> BuiltinResult<(Vec<usize>, Vec<f64>)> {
    match value {
        Value::Tensor(t) => Ok((t.shape.clone(), t.data.clone())),
        Value::Num(n) => Ok((vec![1, 1], vec![*n])),
        Value::Int(i) => Ok((vec![1, 1], vec![i.to_i64() as f64])),
        _ => Err(data_error(
            "DataArray.write supports tensor or numeric scalar values",
        )),
    }
}

#[derive(Clone, Copy, Debug)]
struct DimRange {
    start: usize,
    end: usize,
}

fn read_slice_payload(
    payload: &DataArrayPayload,
    slice_spec: &Value,
) -> BuiltinResult<DataArrayPayload> {
    let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
    let out_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let mut out_values = Vec::new();
    let mut out_index = vec![0usize; out_shape.len()];
    loop {
        let source_index: Vec<usize> = out_index
            .iter()
            .enumerate()
            .map(|(dim, idx)| ranges[dim].start + *idx)
            .collect();
        let linear = linear_index_column_major(&source_index, &payload.shape)?;
        out_values.push(payload.values[linear]);

        if !advance_index(&mut out_index, &out_shape) {
            break;
        }
    }
    Ok(DataArrayPayload {
        dtype: payload.dtype.clone(),
        shape: out_shape,
        values: out_values,
    })
}

fn write_slice_payload(
    payload: &DataArrayPayload,
    slice_spec: &Value,
    rhs: &Value,
) -> BuiltinResult<DataArrayPayload> {
    let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
    let target_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let (rhs_shape, rhs_values) = value_to_tensor_shape_values(rhs)?;
    if rhs_shape != target_shape {
        return Err(data_error(format!(
            "SHAPE_MISMATCH: rhs shape {:?} must match target slice shape {:?}",
            rhs_shape, target_shape
        )));
    }

    let mut next = payload.values.clone();
    let mut rhs_index = vec![0usize; target_shape.len()];
    let mut rhs_linear = 0usize;
    loop {
        let target_index: Vec<usize> = rhs_index
            .iter()
            .enumerate()
            .map(|(dim, idx)| ranges[dim].start + *idx)
            .collect();
        let target_linear = linear_index_column_major(&target_index, &payload.shape)?;
        next[target_linear] = rhs_values[rhs_linear];
        rhs_linear += 1;

        if !advance_index(&mut rhs_index, &target_shape) {
            break;
        }
    }

    Ok(DataArrayPayload {
        dtype: payload.dtype.clone(),
        shape: payload.shape.clone(),
        values: next,
    })
}

fn parse_slice_spec(slice_spec: &Value, shape: &[usize]) -> BuiltinResult<Vec<DimRange>> {
    match slice_spec {
        Value::Cell(cell) => {
            if cell.data.is_empty() {
                return Err(data_error("INVALID_SLICE: empty slice specification"));
            }
            let mut ranges = Vec::with_capacity(shape.len());
            for (dim, extent) in shape.iter().enumerate() {
                if let Some(item) = cell.data.get(dim).map(|v| &**v) {
                    ranges.push(parse_dim_range(item, *extent)?);
                } else {
                    ranges.push(DimRange {
                        start: 0,
                        end: *extent,
                    });
                }
            }
            Ok(ranges)
        }
        Value::String(s) if s == ":" => Ok(shape
            .iter()
            .map(|extent| DimRange {
                start: 0,
                end: *extent,
            })
            .collect()),
        _ => Err(data_error(
            "INVALID_SLICE: slice must be a cell spec like {1:10, :} or ':'",
        )),
    }
}

fn parse_dim_range(value: &Value, extent: usize) -> BuiltinResult<DimRange> {
    if extent == 0 {
        return Ok(DimRange { start: 0, end: 0 });
    }
    match value {
        Value::String(s) if s == ":" => Ok(DimRange {
            start: 0,
            end: extent,
        }),
        Value::Num(n) => {
            let idx = (*n as isize) - 1;
            if idx < 0 || idx as usize >= extent {
                return Err(data_error("INVALID_SLICE: index out of bounds"));
            }
            Ok(DimRange {
                start: idx as usize,
                end: idx as usize + 1,
            })
        }
        Value::Int(i) => {
            let idx = i.to_i64() - 1;
            if idx < 0 || idx as usize >= extent {
                return Err(data_error("INVALID_SLICE: index out of bounds"));
            }
            Ok(DimRange {
                start: idx as usize,
                end: idx as usize + 1,
            })
        }
        Value::Tensor(t) if t.data.len() == 2 => {
            let start = (t.data[0] as isize) - 1;
            let end_inclusive = (t.data[1] as isize) - 1;
            if start < 0 || end_inclusive < start || end_inclusive as usize >= extent {
                return Err(data_error("INVALID_SLICE: range out of bounds"));
            }
            Ok(DimRange {
                start: start as usize,
                end: end_inclusive as usize + 1,
            })
        }
        _ => Err(data_error(
            "INVALID_SLICE: dimension must be ':', scalar index, or [start end] range",
        )),
    }
}

fn linear_index_column_major(index: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if index.len() != shape.len() {
        return Err(data_error("INVALID_SLICE: rank mismatch"));
    }
    let mut stride = 1usize;
    let mut linear = 0usize;
    for (idx, extent) in index.iter().zip(shape.iter()) {
        if *idx >= *extent {
            return Err(data_error("INVALID_SLICE: index out of bounds"));
        }
        linear += idx * stride;
        stride = stride.saturating_mul(*extent);
    }
    Ok(linear)
}

fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    for dim in 0..shape.len() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return true;
        }
        index[dim] = 0;
    }
    false
}

fn chunk_key(coords: &[usize]) -> String {
    coords
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

fn chunk_start_for_coords(coords: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    coords
        .iter()
        .enumerate()
        .map(|(dim, coord)| coord * chunk_shape.get(dim).copied().unwrap_or(1).max(1))
        .collect()
}

fn chunk_extent_for_start(start: &[usize], chunk_shape: &[usize], shape: &[usize]) -> Vec<usize> {
    start
        .iter()
        .enumerate()
        .map(|(dim, start)| {
            let chunk = chunk_shape.get(dim).copied().unwrap_or(1).max(1);
            let end = (*start + chunk).min(shape[dim]);
            end.saturating_sub(*start)
        })
        .collect()
}

fn chunk_intersection(
    ranges: &[DimRange],
    chunk_start: &[usize],
    chunk_extent: &[usize],
) -> Vec<DimRange> {
    let mut out = Vec::with_capacity(ranges.len());
    for dim in 0..ranges.len() {
        let c_start = chunk_start[dim];
        let c_end = c_start + chunk_extent[dim];
        let start = ranges[dim].start.max(c_start);
        let end = ranges[dim].end.min(c_end);
        if start >= end {
            return Vec::new();
        }
        out.push(DimRange { start, end });
    }
    out
}

fn touched_chunk_coords(
    ranges: &[DimRange],
    chunk_shape: &[usize],
    shape: &[usize],
) -> Vec<Vec<usize>> {
    let mut span = Vec::with_capacity(ranges.len());
    let mut begin = Vec::with_capacity(ranges.len());
    for dim in 0..ranges.len() {
        if shape[dim] == 0 {
            return Vec::new();
        }
        let chunk = chunk_shape.get(dim).copied().unwrap_or(1).max(1);
        let first = ranges[dim].start / chunk;
        let last = (ranges[dim].end.saturating_sub(1)) / chunk;
        begin.push(first);
        span.push(last.saturating_sub(first) + 1);
    }
    let mut local = vec![0usize; span.len()];
    let mut out = Vec::new();
    loop {
        out.push(
            local
                .iter()
                .enumerate()
                .map(|(dim, v)| begin[dim] + *v)
                .collect::<Vec<_>>(),
        );
        if !advance_index(&mut local, &span) {
            break;
        }
    }
    out
}

async fn maybe_upload_chunk_batch_async(
    root: &std::path::Path,
    array_name: &str,
    batch: Vec<(DataChunkDescriptor, Vec<u8>)>,
) -> BuiltinResult<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let request = DataChunkUploadRequest {
        dataset_path: root.to_string_lossy().to_string(),
        array: array_name.to_string(),
        chunks: batch.iter().map(|(d, _)| d.clone()).collect(),
    };
    let targets = match runmat_filesystem::data_chunk_upload_targets_async(&request).await {
        Ok(targets) => targets,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => return Ok(()),
        Err(err) => {
            return Err(data_error(format!(
                "failed to request data chunk upload targets: {err}"
            )))
        }
    };
    for (descriptor, bytes) in batch {
        let target = targets
            .iter()
            .find(|t| t.key == descriptor.key)
            .ok_or_else(|| {
                data_error(format!(
                    "missing upload target for chunk '{}'",
                    descriptor.key
                ))
            })?;
        runmat_filesystem::data_upload_chunk_async(target, &bytes)
            .await
            .map_err(|err| {
                data_error(format!(
                    "failed to upload chunk '{}': {err}",
                    descriptor.key
                ))
            })?;
        tracing::info!(
            target: "runmat.data",
            dataset = %root.display(),
            array = array_name,
            chunk_key = descriptor.key,
            bytes = bytes.len(),
            "chunk upload completed"
        );
    }
    Ok(())
}

fn chunk_rel_path(array_name: &str, object_id: &str) -> String {
    format!("arrays/{array_name}/chunks/{object_id}.json")
}

async fn load_or_init_chunk(
    root: &std::path::Path,
    array_name: &str,
    key: &str,
    coords: &[usize],
    chunk_extent: &[usize],
    pos_by_key: &HashMap<String, usize>,
    chunk_index: &DataChunkIndex,
) -> BuiltinResult<(usize, bool, DataChunkIndexEntry, DataArrayPayload)> {
    if let Some(index) = pos_by_key.get(key).copied() {
        let entry = chunk_index
            .chunks
            .get(index)
            .cloned()
            .ok_or_else(|| data_error(format!("chunk index missing key '{key}'")))?;
        let bytes = runmat_filesystem::read_async(root.join(&entry.data_path))
            .await
            .map_err(|err| {
                data_error(format!(
                    "failed to read chunk payload '{}': {err}",
                    entry.data_path
                ))
            })?;
        let payload: DataArrayPayload = serde_json::from_slice(&bytes).map_err(|err| {
            data_error(format!(
                "failed to parse chunk payload '{}': {err}",
                entry.data_path
            ))
        })?;
        return Ok((index, true, entry, payload));
    }

    let object_id = format!("obj_{}", key.replace('.', "_"));
    let entry = DataChunkIndexEntry {
        key: key.to_string(),
        object_id: object_id.clone(),
        hash: String::new(),
        bytes_raw: 0,
        bytes_stored: 0,
        coords: coords.to_vec(),
        shape: chunk_extent.to_vec(),
        data_path: chunk_rel_path(array_name, &object_id),
    };
    let payload = DataArrayPayload {
        dtype: "f64".to_string(),
        shape: chunk_extent.to_vec(),
        values: vec![0.0; chunk_extent.iter().copied().product()],
    };
    Ok((chunk_index.chunks.len(), false, entry, payload))
}

fn attrs_to_struct(attrs: &BTreeMap<String, serde_json::Value>) -> Value {
    let mut out = StructValue::new();
    for (k, v) in attrs {
        out.fields.insert(k.clone(), json_to_value(v));
    }
    Value::Struct(out)
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::CharArray(ca) => serde_json::Value::String(ca.to_string()),
        Value::Num(n) => serde_json::json!(n),
        Value::Int(i) => serde_json::json!(i.to_i64()),
        Value::Bool(b) => serde_json::json!(b),
        _ => serde_json::Value::String(format!("{value:?}")),
    }
}

fn json_to_value(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => Value::Num(n.as_f64().unwrap_or_default()),
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let vals = arr.iter().map(json_to_value).collect::<Vec<_>>();
            crate::make_cell(vals.clone(), 1, vals.len())
                .unwrap_or_else(|_| Value::String("<invalid-array>".to_string()))
        }
        serde_json::Value::Object(map) => {
            let mut s = StructValue::new();
            for (k, v) in map {
                s.fields.insert(k.clone(), json_to_value(v));
            }
            Value::Struct(s)
        }
        serde_json::Value::Null => Value::String("".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::call_builtin;
    use async_trait::async_trait;
    use axum::extract::{Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::{post, put};
    use axum::{Json, Router};
    use runmat_builtins::CellArray;
    use runmat_filesystem::data_contract::{
        DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
    };
    use runmat_filesystem::{
        DirEntry, FileHandle, FsMetadata, FsProvider, NativeFsProvider, OpenFlags,
    };
    use serde::Deserialize;
    use std::path::Path;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
    use tokio::runtime::Runtime;
    use tokio::sync::oneshot;

    fn serial_test_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("data test serial lock poisoned")
    }

    #[derive(Default)]
    struct CountingDataUploadProvider {
        inner: NativeFsProvider,
        uploaded_keys: Arc<Mutex<Vec<String>>>,
    }

    struct HttpDataUploadProvider {
        inner: NativeFsProvider,
        base_url: String,
        client: reqwest::blocking::Client,
    }

    impl HttpDataUploadProvider {
        fn new(base_url: String) -> Self {
            Self {
                inner: NativeFsProvider,
                base_url,
                client: reqwest::blocking::Client::new(),
            }
        }
    }

    #[async_trait(?Send)]
    impl FsProvider for HttpDataUploadProvider {
        fn open(&self, path: &Path, flags: &OpenFlags) -> std::io::Result<Box<dyn FileHandle>> {
            self.inner.open(path, flags)
        }

        async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            self.inner.read(path).await
        }

        async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
            self.inner.write(path, data).await
        }

        async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_file(path).await
        }

        async fn metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
            self.inner.metadata(path).await
        }

        async fn symlink_metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
            self.inner.symlink_metadata(path).await
        }

        async fn read_dir(&self, path: &Path) -> std::io::Result<Vec<DirEntry>> {
            self.inner.read_dir(path).await
        }

        async fn canonicalize(&self, path: &Path) -> std::io::Result<std::path::PathBuf> {
            self.inner.canonicalize(path).await
        }

        async fn create_dir(&self, path: &Path) -> std::io::Result<()> {
            self.inner.create_dir(path).await
        }

        async fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
            self.inner.create_dir_all(path).await
        }

        async fn remove_dir(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_dir(path).await
        }

        async fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_dir_all(path).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            self.inner.rename(from, to).await
        }

        async fn set_readonly(&self, path: &Path, readonly: bool) -> std::io::Result<()> {
            self.inner.set_readonly(path, readonly).await
        }

        async fn data_manifest_descriptor(
            &self,
            request: &DataManifestRequest,
        ) -> std::io::Result<DataManifestDescriptor> {
            self.inner.data_manifest_descriptor(request).await
        }

        async fn data_chunk_upload_targets(
            &self,
            request: &DataChunkUploadRequest,
        ) -> std::io::Result<Vec<DataChunkUploadTarget>> {
            #[derive(Deserialize)]
            struct UploadTargetsResponse {
                targets: Vec<DataChunkUploadTarget>,
            }
            let url = format!("{}/data/chunks/upload-targets", self.base_url);
            let response = self
                .client
                .post(url)
                .json(request)
                .send()
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            if !response.status().is_success() {
                return Err(std::io::Error::other(format!(
                    "upload targets request failed: {}",
                    response.status()
                )));
            }
            let parsed: UploadTargetsResponse = response
                .json()
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            Ok(parsed.targets)
        }

        async fn data_upload_chunk(
            &self,
            target: &DataChunkUploadTarget,
            data: &[u8],
        ) -> std::io::Result<()> {
            let upload_url = if let Some(key) = target.upload_url.strip_prefix("upload://") {
                format!("{}/upload?key={}", self.base_url, key)
            } else {
                target.upload_url.clone()
            };
            let method = reqwest::Method::from_bytes(target.method.as_bytes())
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            let mut request = self.client.request(method, &upload_url);
            for (k, v) in &target.headers {
                request = request.header(k, v);
            }
            let response = request
                .body(data.to_vec())
                .send()
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            if !response.status().is_success() {
                return Err(std::io::Error::other(format!(
                    "chunk upload failed: {}",
                    response.status()
                )));
            }
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct UploadHarness {
        uploads: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Deserialize)]
    struct UploadChunkQuery {
        key: String,
    }

    async fn upload_targets_handler(
        Json(req): Json<DataChunkUploadRequest>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let targets = req
            .chunks
            .iter()
            .map(|chunk| {
                serde_json::json!({
                    "key": chunk.key,
                    "method": "PUT",
                    "upload_url": format!("upload://{}", chunk.key),
                    "headers": {
                        "x-runmat-hash": chunk.hash,
                    }
                })
            })
            .collect::<Vec<_>>();
        Ok(Json(serde_json::json!({ "targets": targets })))
    }

    async fn upload_handler(
        State(harness): State<UploadHarness>,
        Query(query): Query<UploadChunkQuery>,
        headers: HeaderMap,
        body: axum::body::Bytes,
    ) -> Result<(), StatusCode> {
        if body.is_empty() {
            return Err(StatusCode::BAD_REQUEST);
        }
        if headers.get("x-runmat-hash").is_none() {
            return Err(StatusCode::BAD_REQUEST);
        }
        let mut guard = harness.uploads.lock().expect("uploads lock poisoned");
        guard.push(query.key);
        Ok(())
    }

    fn spawn_upload_server() -> (
        String,
        Arc<Mutex<Vec<String>>>,
        Runtime,
        oneshot::Sender<()>,
    ) {
        let harness = UploadHarness::default();
        let uploads = Arc::clone(&harness.uploads);
        let runtime = Runtime::new().expect("tokio runtime");
        let (addr, shutdown_tx) = runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
                .await
                .expect("bind upload server");
            let addr = listener.local_addr().expect("local addr");
            let app = Router::new()
                .route("/data/chunks/upload-targets", post(upload_targets_handler))
                .route("/upload", put(upload_handler))
                .with_state(harness);
            let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
            let server = axum::serve(listener, app).with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            });
            tokio::spawn(async move {
                let _ = server.await;
            });
            (addr, shutdown_tx)
        });
        (format!("http://{}", addr), uploads, runtime, shutdown_tx)
    }

    impl CountingDataUploadProvider {
        fn uploaded_keys(&self) -> Arc<Mutex<Vec<String>>> {
            Arc::clone(&self.uploaded_keys)
        }
    }

    #[async_trait(?Send)]
    impl FsProvider for CountingDataUploadProvider {
        fn open(&self, path: &Path, flags: &OpenFlags) -> std::io::Result<Box<dyn FileHandle>> {
            self.inner.open(path, flags)
        }

        async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            self.inner.read(path).await
        }

        async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
            self.inner.write(path, data).await
        }

        async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_file(path).await
        }

        async fn metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
            self.inner.metadata(path).await
        }

        async fn symlink_metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
            self.inner.symlink_metadata(path).await
        }

        async fn read_dir(&self, path: &Path) -> std::io::Result<Vec<DirEntry>> {
            self.inner.read_dir(path).await
        }

        async fn canonicalize(&self, path: &Path) -> std::io::Result<std::path::PathBuf> {
            self.inner.canonicalize(path).await
        }

        async fn create_dir(&self, path: &Path) -> std::io::Result<()> {
            self.inner.create_dir(path).await
        }

        async fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
            self.inner.create_dir_all(path).await
        }

        async fn remove_dir(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_dir(path).await
        }

        async fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_dir_all(path).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            self.inner.rename(from, to).await
        }

        async fn set_readonly(&self, path: &Path, readonly: bool) -> std::io::Result<()> {
            self.inner.set_readonly(path, readonly).await
        }

        async fn data_manifest_descriptor(
            &self,
            request: &DataManifestRequest,
        ) -> std::io::Result<DataManifestDescriptor> {
            self.inner.data_manifest_descriptor(request).await
        }

        async fn data_chunk_upload_targets(
            &self,
            request: &DataChunkUploadRequest,
        ) -> std::io::Result<Vec<DataChunkUploadTarget>> {
            Ok(request
                .chunks
                .iter()
                .map(|chunk| DataChunkUploadTarget {
                    key: chunk.key.clone(),
                    method: "PUT".to_string(),
                    upload_url: format!("count://{}", chunk.object_id),
                    headers: std::collections::HashMap::new(),
                })
                .collect())
        }

        async fn data_upload_chunk(
            &self,
            target: &DataChunkUploadTarget,
            _data: &[u8],
        ) -> std::io::Result<()> {
            let mut guard = match self.uploaded_keys.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.push(target.key.clone());
            Ok(())
        }
    }

    #[test]
    fn create_open_write_read_dataset() {
        let _serial = serial_test_guard();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.data").to_string_lossy().to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(runmat_builtins::CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");

        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");
        let write_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("write tensor");
        call_builtin(
            "DataArray.write",
            &[arr.clone(), Value::Tensor(write_tensor)],
        )
        .expect("write array");

        let read_back = call_builtin("DataArray.read", &[arr]).expect("read array");
        let Value::Tensor(t) = read_back else {
            panic!("expected tensor");
        };
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn write_and_read_slice_payload() {
        let _serial = serial_test_guard();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("slice.data").to_string_lossy().to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![3.0, 3.0], vec![1, 2]).expect("shape tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");

        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                    Value::String(":".to_string()),
                ],
                1,
                2,
            )
            .expect("slice cell"),
        );
        let rhs = Value::Tensor(
            Tensor::new(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0], vec![2, 3]).expect("rhs"),
        );
        call_builtin("DataArray.write", &[arr.clone(), slice.clone(), rhs]).expect("slice write");

        let read_back = call_builtin("DataArray.read", &[arr.clone(), slice]).expect("slice read");
        let Value::Tensor(t) = read_back else {
            panic!("expected tensor");
        };
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn slice_write_updates_only_touched_chunks() {
        let _serial = serial_test_guard();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("chunked.data")
            .to_string_lossy()
            .to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
        );
        array_meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");

        let full = Value::Tensor(
            Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).expect("full tensor"),
        );
        call_builtin("DataArray.write", &[arr.clone(), full]).expect("initial write");

        let root = std::path::PathBuf::from(&path);
        let untouched_path = root.join("arrays/temperature/chunks/obj_1_1.json");
        let touched_path = root.join("arrays/temperature/chunks/obj_0_0.json");
        let untouched_before =
            futures::executor::block_on(runmat_filesystem::read_async(&untouched_path))
                .expect("read untouched before");
        let touched_before =
            futures::executor::block_on(runmat_filesystem::read_async(&touched_path))
                .expect("read touched before");

        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                ],
                1,
                2,
            )
            .expect("slice cell"),
        );
        let rhs =
            Value::Tensor(Tensor::new(vec![99.0, 98.0, 97.0, 96.0], vec![2, 2]).expect("rhs"));
        call_builtin("DataArray.write", &[arr.clone(), slice, rhs]).expect("slice write");

        let untouched_after =
            futures::executor::block_on(runmat_filesystem::read_async(&untouched_path))
                .expect("read untouched after");
        let touched_after =
            futures::executor::block_on(runmat_filesystem::read_async(&touched_path))
                .expect("read touched after");
        assert_eq!(untouched_before, untouched_after);
        assert_ne!(touched_before, touched_after);
    }

    #[test]
    fn slice_write_uploads_only_touched_chunk_targets() {
        let _serial = serial_test_guard();
        let provider = Arc::new(CountingDataUploadProvider::default());
        let uploaded = provider.uploaded_keys();
        let _guard = runmat_filesystem::replace_provider(provider);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("remote-chunked.data")
            .to_string_lossy()
            .to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
        );
        array_meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");

        call_builtin(
            "DataArray.write",
            &[
                arr.clone(),
                Value::Tensor(
                    Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4])
                        .expect("full tensor"),
                ),
            ],
        )
        .expect("initial write");

        let manifest =
            futures::executor::block_on(crate::data::read_manifest_async(&dataset_root(&path)))
                .expect("manifest after initial write");
        let meta = manifest
            .arrays
            .get("temperature")
            .expect("temperature meta");
        let chunk_index_path =
            dataset_root(&path).join(meta.chunk_index_path.clone().expect("chunk index path"));
        assert!(
            futures::executor::block_on(runmat_filesystem::metadata_async(&chunk_index_path))
                .is_ok()
        );

        {
            let mut keys = uploaded.lock().expect("uploaded keys lock");
            keys.clear();
        }

        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                ],
                1,
                2,
            )
            .expect("slice cell"),
        );
        let rhs = Value::Tensor(Tensor::new(vec![9.0, 8.0, 7.0, 6.0], vec![2, 2]).expect("rhs"));
        call_builtin("DataArray.write", &[arr, slice, rhs]).expect("slice write");

        let keys = uploaded.lock().expect("uploaded keys lock");
        assert_eq!(keys.as_slice(), ["0.0".to_string()].as_slice());
    }

    #[test]
    fn slice_write_uploads_expected_cross_boundary_chunk_targets() {
        let _serial = serial_test_guard();
        let provider = Arc::new(CountingDataUploadProvider::default());
        let uploaded = provider.uploaded_keys();
        let _guard = runmat_filesystem::replace_provider(provider);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("remote-chunked-boundary.data")
            .to_string_lossy()
            .to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
        );
        array_meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");

        call_builtin(
            "DataArray.write",
            &[
                arr.clone(),
                Value::Tensor(
                    Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4])
                        .expect("full tensor"),
                ),
            ],
        )
        .expect("initial write");
        {
            let mut keys = uploaded.lock().expect("uploaded keys lock");
            keys.clear();
        }

        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                    Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                ],
                1,
                2,
            )
            .expect("slice cell"),
        );
        let rhs =
            Value::Tensor(Tensor::new(vec![19.0, 18.0, 17.0, 16.0], vec![2, 2]).expect("rhs"));
        call_builtin("DataArray.write", &[arr, slice, rhs]).expect("slice write");

        let mut keys = uploaded.lock().expect("uploaded keys lock").clone();
        keys.sort();
        keys.dedup();
        assert_eq!(
            keys.as_slice(),
            [
                "0.0".to_string(),
                "0.1".to_string(),
                "1.0".to_string(),
                "1.1".to_string(),
            ]
            .as_slice()
        );
    }

    #[test]
    fn slice_write_hits_http_server_data_endpoints_with_expected_keys() {
        let _serial = serial_test_guard();
        let (base_url, uploads, runtime, shutdown_tx) = spawn_upload_server();
        let provider = Arc::new(HttpDataUploadProvider::new(base_url));
        let _guard = runmat_filesystem::replace_provider(provider);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("http-endpoint.data")
            .to_string_lossy()
            .to_string();

        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
        );
        array_meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
        );
        let mut arrays = StructValue::new();
        arrays
            .fields
            .insert("temperature".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("temperature".to_string())],
        )
        .expect("dataset array");

        call_builtin(
            "DataArray.write",
            &[
                arr.clone(),
                Value::Tensor(
                    Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4])
                        .expect("full tensor"),
                ),
            ],
        )
        .expect("initial write");

        {
            let mut keys = uploads.lock().expect("uploads lock");
            keys.clear();
        }

        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                    Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                ],
                1,
                2,
            )
            .expect("slice cell"),
        );
        let rhs =
            Value::Tensor(Tensor::new(vec![19.0, 18.0, 17.0, 16.0], vec![2, 2]).expect("rhs"));
        call_builtin("DataArray.write", &[arr, slice, rhs]).expect("slice write");

        let mut keys = uploads.lock().expect("uploads lock").clone();
        keys.sort();
        keys.dedup();
        assert_eq!(
            keys.as_slice(),
            [
                "0.0".to_string(),
                "0.1".to_string(),
                "1.0".to_string(),
                "1.1".to_string(),
            ]
            .as_slice()
        );

        let _ = shutdown_tx.send(());
        drop(runtime);
    }

    #[test]
    fn tx_create_resize_fill_and_delete_array() {
        let _serial = serial_test_guard();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tx-ops.data").to_string_lossy().to_string();

        let mut arrays = StructValue::new();
        let mut array_meta = StructValue::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("shape tensor")),
        );
        arrays
            .fields
            .insert("base".to_string(), Value::Struct(array_meta));
        let mut schema = StructValue::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays));

        let ds = call_builtin(
            "data.create",
            &[
                Value::String(path.clone()),
                Value::Struct(schema),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("create dataset");

        let tx = call_builtin("Dataset.begin", &[ds]).expect("begin tx");
        let mut new_meta = StructValue::new();
        new_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string()));
        new_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape tensor")),
        );
        call_builtin(
            "DataTransaction.create_array",
            &[
                tx.clone(),
                Value::String("new_array".to_string()),
                Value::Struct(new_meta),
            ],
        )
        .expect("create array in tx");
        call_builtin(
            "DataTransaction.resize",
            &[
                tx.clone(),
                Value::String("new_array".to_string()),
                Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![1, 2]).expect("shape tensor")),
            ],
        )
        .expect("resize array in tx");
        call_builtin(
            "DataTransaction.fill",
            &[
                tx.clone(),
                Value::String("new_array".to_string()),
                Value::Num(7.0),
            ],
        )
        .expect("fill array in tx");
        call_builtin(
            "DataTransaction.delete_array",
            &[tx.clone(), Value::String("base".to_string())],
        )
        .expect("delete array in tx");
        call_builtin("DataTransaction.commit", &[tx]).expect("commit tx");

        let ds = call_builtin(
            "data.open",
            &[
                Value::String(path),
                Value::Cell(CellArray::new(vec![], 1, 0).expect("cell")),
            ],
        )
        .expect("open dataset");
        let has_base = call_builtin(
            "Dataset.has_array",
            &[ds.clone(), Value::String("base".to_string())],
        )
        .expect("has base");
        assert_eq!(has_base, Value::Bool(false));
        let arr = call_builtin(
            "Dataset.array",
            &[ds, Value::String("new_array".to_string())],
        )
        .expect("new array");
        let read_back = call_builtin("DataArray.read", &[arr]).expect("read array");
        let Value::Tensor(t) = read_back else {
            panic!("expected tensor");
        };
        assert_eq!(t.shape, vec![3, 1]);
        assert_eq!(t.data, vec![7.0, 7.0, 7.0]);
    }
}
