//! Managed data receiver runtime bindings.

use super::*;
use crate::context::{host_effect::DatasetHandleOperation as Op, RuntimeContext};

#[runtime_builtin(
    name = "DataArray.name",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_name_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::ArrayName, &[], async {
        Ok(Value::string(name))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.dtype",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_dtype_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    metadata(runtime.as_ref(), &owner, &name, Op::ArrayDType, |meta| {
        Ok(Value::string(meta.dtype.clone()))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.shape",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_shape_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    metadata(runtime.as_ref(), &owner, &name, Op::ArrayShape, |meta| {
        let values = meta.shape.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let tensor = Tensor::new(values, vec![1, meta.shape.len()])
            .map_err(|err| data_error(format!("DataArray.shape: {err}")))?;
        Ok(Value::Tensor(tensor))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.rank",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_rank_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    metadata(runtime.as_ref(), &owner, &name, Op::ArrayRank, |meta| {
        Ok(Value::Num(meta.shape.len() as f64))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.chunk_shape",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_chunk_shape_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    metadata(
        runtime.as_ref(),
        &owner,
        &name,
        Op::ArrayChunkShape,
        |meta| {
            let values = meta
                .chunk_shape
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>();
            let tensor = Tensor::new(values, vec![1, meta.chunk_shape.len()])
                .map_err(|err| data_error(format!("DataArray.chunk_shape: {err}")))?;
            Ok(Value::Tensor(tensor))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataArray.codec",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_codec_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    metadata(runtime.as_ref(), &owner, &name, Op::ArrayCodec, |meta| {
        Ok(Value::string(meta.codec.clone()))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.read",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_read_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(data_error("DataArray.read expects at most one slice"));
    }
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    let inputs = rest.iter().collect::<Vec<_>>();
    handle_admission::execute(runtime.as_ref(), &owner, Op::ArrayRead, &inputs, async {
        let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        let meta = manifest
            .arrays
            .get(&name)
            .ok_or_else(|| data_error(format!("DataArray.read: array '{name}' not found")))?;
        let payload = crate::data::read_array_payload_with_budget_on(
            owner.filesystem.as_ref(),
            &owner.root,
            meta,
            &mut budget,
        )
        .await?;
        let sliced = if let Some(slice_spec) = rest.first() {
            read_slice_payload(&payload, slice_spec, &mut budget).await?
        } else {
            payload
        };
        sliced
            .into_value()
            .map_err(|err| data_error(format!("DataArray.read: {err}")))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.write",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_write_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    let (slice_spec, value) = match (rest.len(), rest.get(0), rest.get(1)) {
        (1, Some(v), _) => (None, v),
        (2, Some(slice), Some(v)) => (Some(slice), v),
        _ => {
            return Err(data_error(
                "DataArray.write expects values or (sliceSpec, values) arguments",
            ))
        }
    };
    let inputs = rest.iter().collect::<Vec<_>>();
    handle_admission::execute(runtime.as_ref(), &owner, Op::ArrayWrite, &inputs, async {
        let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
        write_array_full_on(
            owner.filesystem.as_ref(),
            &owner.root,
            &name,
            slice_spec,
            value,
            &mut budget,
        )
        .await?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.resize",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_resize_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    new_shape: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(data_error(
            "DataArray.resize does not accept trailing options",
        ));
    }
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    let shape = parse_shape_from_value(&new_shape)?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::ArrayResize,
        &[&new_shape],
        async {
            let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
            let mut manifest =
                crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
            let meta = manifest
                .arrays
                .get_mut(&name)
                .ok_or_else(|| data_error(format!("DataArray.resize: array '{name}' not found")))?;
            meta.shape = shape.clone();
            crate::data::cooperate_data_work(&mut budget, 0).await?;
            let payload =
                DataArrayPayload::zeros_cooperative(meta.dtype.clone(), shape.clone(), &mut budget)
                    .await?;
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
            crate::data::write_manifest_on(owner.filesystem.as_ref(), &owner.root, &manifest)
                .await?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

async fn metadata(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    name: &str,
    operation: Op,
    project: impl FnOnce(&DataArrayMeta) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    handle_admission::execute(runtime, owner, operation, &[], async {
        let manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        let meta = manifest.arrays.get(name).ok_or_else(|| {
            data_error(format!(
                "{}: array '{name}' not found",
                operation.public_name()
            ))
        })?;
        project(meta)
    })
    .await
}

#[runtime_builtin(
    name = "DataArray.fill",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::array"
)]
async fn data_array_fill_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(data_error(
            "DataArray.fill does not accept trailing options",
        ));
    }
    let (owner, name) = authority::resolve_array(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::ArrayFill, &[&value], async {
        let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
        let mut manifest =
            crate::data::read_manifest_on(owner.filesystem.as_ref(), &owner.root).await?;
        let meta = manifest
            .arrays
            .get_mut(&name)
            .ok_or_else(|| data_error(format!("DataArray.fill: array '{name}' not found")))?;
        crate::data::cooperate_data_work(&mut budget, 0).await?;
        let payload = DataArrayPayload::filled_cooperative(
            meta.dtype.clone(),
            meta.shape.clone(),
            &value,
            &mut budget,
        )
        .await?;
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
    })
    .await
}
