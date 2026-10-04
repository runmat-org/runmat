//! Managed data receiver runtime bindings.

use super::*;

#[runtime_builtin(
    name = "DataArray.name",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_name_builtin(base: Value) -> BuiltinResult<Value> {
    let (_, name) = authority::resolve_array(&base)?;
    Ok(Value::String(name))
}

#[runtime_builtin(
    name = "DataArray.dtype",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_dtype_builtin(base: Value) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.dtype: array '{name}' not found")))?;
    Ok(Value::String(meta.dtype.clone()))
}

#[runtime_builtin(
    name = "DataArray.shape",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_shape_builtin(base: Value) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_rank_builtin(base: Value) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.rank: array '{name}' not found")))?;
    Ok(Value::Num(meta.shape.len() as f64))
}

#[runtime_builtin(
    name = "DataArray.chunk_shape",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_chunk_shape_builtin(base: Value) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_codec_builtin(base: Value) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.codec: array '{name}' not found")))?;
    Ok(Value::String(meta.codec.clone()))
}

#[runtime_builtin(
    name = "DataArray.read",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_read_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let manifest = crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get(&name)
        .ok_or_else(|| data_error(format!("DataArray.read: array '{name}' not found")))?;
    let payload =
        crate::data::read_array_payload_on(owner.filesystem.as_ref(), &owner.root, meta).await?;
    let sliced = if let Some(slice_spec) = rest.first() {
        read_slice_payload(&payload, slice_spec)?
    } else {
        payload
    };
    sliced
        .into_value()
        .map_err(|err| data_error(format!("DataArray.read: {err}")))
}

#[runtime_builtin(
    name = "DataArray.write",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_write_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let (slice_spec, value) = match rest.as_slice() {
        [v] => (None, v),
        [slice, v] => (Some(slice), v),
        _ => {
            return Err(data_error(
                "DataArray.write expects values or (sliceSpec, values) arguments",
            ))
        }
    };
    write_array_full_on(
        owner.filesystem.as_ref(),
        &owner.root,
        &name,
        slice_spec,
        value,
    )
    .await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataArray.resize",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_resize_builtin(
    base: Value,
    new_shape: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let shape = parse_shape_from_value(&new_shape)?;
    let mut manifest =
        crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get_mut(&name)
        .ok_or_else(|| data_error(format!("DataArray.resize: array '{name}' not found")))?;
    meta.shape = shape.clone();
    let payload = DataArrayPayload::zeros(meta.dtype.clone(), shape.clone())?;
    let mut budget = None;
    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        owner.filesystem.as_ref(),
        &owner.root,
        &name,
        &payload,
        &meta.chunk_shape,
        &mut budget,
    )
    .await?;
    meta.data_path = make_rel_data_path(&owner.root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(&owner.root, &chunk_index_path)?);
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest).await?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataArray.fill",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_array_fill_builtin(
    base: Value,
    value: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(&base)?;
    let mut manifest =
        crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
    let meta = manifest
        .arrays
        .get_mut(&name)
        .ok_or_else(|| data_error(format!("DataArray.fill: array '{name}' not found")))?;
    let payload = DataArrayPayload::filled(meta.dtype.clone(), meta.shape.clone(), &value)?;
    let mut budget = None;
    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        owner.filesystem.as_ref(),
        &owner.root,
        &name,
        &payload,
        &meta.chunk_shape,
        &mut budget,
    )
    .await?;
    meta.data_path = make_rel_data_path(&owner.root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(&owner.root, &chunk_index_path)?);
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest).await?;
    Ok(Value::Bool(true))
}
