//! Dataset manifest domain helpers.

use super::*;

pub(in crate::builtins::io::data) async fn create_array_in_manifest(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    mut meta: DataArrayMeta,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    if manifest.arrays.contains_key(array_name) {
        return Err(data_error(format!(
            "DataTransaction.create_array: array '{array_name}' already exists"
        )));
    }
    let payload =
        DataArrayPayload::zeros_cooperative(meta.dtype.clone(), meta.shape.clone(), budget).await?;
    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        provider,
        root,
        array_name,
        &payload,
        &meta.chunk_shape,
        budget,
    )
    .await?;
    meta.data_path = make_rel_data_path(root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    manifest.arrays.insert(array_name.to_string(), meta);
    Ok(())
}

pub(in crate::builtins::io::data) async fn resize_array_in_manifest(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    shape: Vec<usize>,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let meta = manifest
        .arrays
        .get_mut(array_name)
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;
    meta.shape = shape.clone();
    let payload =
        DataArrayPayload::zeros_cooperative(meta.dtype.clone(), shape.clone(), budget).await?;
    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        provider,
        root,
        array_name,
        &payload,
        &meta.chunk_shape,
        budget,
    )
    .await?;
    meta.data_path = make_rel_data_path(root, &payload_path)?;
    meta.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    Ok(())
}

pub(in crate::builtins::io::data) async fn fill_array_in_manifest(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let meta: DataArrayMeta = manifest
        .arrays
        .get(array_name)
        .cloned()
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;
    let payload =
        crate::data::read_array_payload_with_budget_on(provider, root, &meta, budget).await?;
    let next_payload = if let Some(slice_spec) = slice_spec {
        let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
        let target_shape: Vec<usize> = ranges
            .iter()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        let rhs = DataArrayPayload::filled_cooperative(
            payload.dtype.clone(),
            target_shape,
            value,
            budget,
        )
        .await?
        .into_value()
        .map_err(|err| data_error(format!("DataTransaction.fill: {err}")))?;
        write_slice_payload(&payload, slice_spec, &rhs, budget).await?
    } else {
        DataArrayPayload::filled_cooperative(
            payload.dtype.clone(),
            payload.shape.clone(),
            value,
            budget,
        )
        .await?
    };
    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        provider,
        root,
        array_name,
        &next_payload,
        &meta.chunk_shape,
        budget,
    )
    .await?;
    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = next_payload.shape.clone();
        updated.data_path = make_rel_data_path(root, &payload_path)?;
        updated.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    }
    Ok(())
}

pub(in crate::builtins::io::data) async fn delete_array_in_manifest_on(
    _provider: &dyn runmat_filesystem::FsProvider,
    _root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let removed = manifest.arrays.remove(array_name);
    if removed.is_none() {
        return Err(data_error(format!(
            "DataTransaction.delete_array: array '{array_name}' not found"
        )));
    }
    // Removing the manifest reference does not authorize deleting storage still
    // reachable through the currently published manifest or a snapshot.
    Ok(())
}

pub(in crate::builtins::io::data) fn parse_array_meta(
    array_name: &str,
    meta: &Value,
) -> BuiltinResult<DataArrayMeta> {
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
    validate_chunk_shape(&shape, &chunk_shape)?;
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

pub(in crate::builtins::io::data) fn default_chunk_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
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

pub(in crate::builtins::io::data) fn parse_shape_from_value(
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => {
            let len = tensor_utils::tensor_element_len(t);
            let mut out = Vec::with_capacity(len);
            for index in 0..len {
                let value = t
                    .numeric_value_at(index)
                    .ok_or_else(|| data_error("shape dimensions require valid numeric storage"))?;
                out.push(numeric_dimension_to_usize(value).ok_or_else(|| {
                    data_error(
                        "shape dimensions must be non-negative finite integers within platform limits",
                    )
                })?);
            }
            Ok(out)
        }
        Value::Num(v) => floating_dimension_to_usize(*v)
            .map(|value| vec![value])
            .ok_or_else(|| {
                data_error(
                    "shape dimensions must be non-negative finite integers within platform limits",
                )
            }),
        Value::Int(v) => v
            .try_to_usize()
            .map(|n| vec![n])
            .ok_or_else(|| data_error("shape dimensions must be non-negative integers")),
        _ => Err(data_error("shape must be a numeric vector")),
    }
}

pub(in crate::builtins::io::data) fn floating_dimension_to_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    let integer = value as u128;
    (integer <= usize::MAX as u128 && integer as f64 == value)
        .then(|| usize::try_from(integer).ok())
        .flatten()
}

pub(in crate::builtins::io::data) fn numeric_dimension_to_usize(
    value: NumericScalar,
) -> Option<usize> {
    match value {
        NumericScalar::F64(value) => floating_dimension_to_usize(value),
        NumericScalar::F32(value) => floating_dimension_to_usize(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_usize()),
    }
}

pub(in crate::builtins::io::data) fn attrs_to_struct(
    attrs: &BTreeMap<String, serde_json::Value>,
) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (k, v) in attrs {
        out.fields.insert(k.clone(), json_to_value(v));
    }
    Value::Struct(out.finish())
}

pub(in crate::builtins::io::data) fn value_to_json(
    value: &Value,
) -> BuiltinResult<serde_json::Value> {
    Ok(match value {
        Value::String(s) => s
            .as_str()
            .map(|s| serde_json::Value::String(s.to_owned()))
            .unwrap_or(serde_json::Value::Null),
        Value::StringArray(strings) if strings.data().len() == 1 => strings.data()[0]
            .as_str()
            .map(|text| serde_json::Value::String(text.to_owned()))
            .unwrap_or(serde_json::Value::Null),
        Value::CharArray(chars) if chars.rows == 1 && chars.shape().iter().skip(2).all(|&extent| extent == 1) => {
            chars.validate_layout().map_err(|cause| data_error(format!("attribute character row has invalid layout: {cause}")))?;
            serde_json::Value::String(crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                data_error("attribute value contains malformed UTF-16 text")
            })?)
        }
        Value::Num(n) if n.is_finite() => serde_json::json!(n),
        Value::Num(_) => return Err(data_error("attribute numeric value must be finite; JSON null is reserved for typed missing strings")),
        Value::Int(i) => int_value_to_json(i),
        Value::Bool(b) => serde_json::json!(b),
        _ => return Err(data_error("attribute value must be a scalar string, character row, finite number, integer, or logical value; aggregate values require explicit encoding")),
    })
}

pub(in crate::builtins::io::data) fn json_to_value(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(value) = n.as_i64() {
                Value::Int(IntValue::I64(value))
            } else if let Some(value) = n.as_u64() {
                Value::Int(IntValue::U64(value))
            } else {
                Value::Num(n.as_f64().unwrap_or_default())
            }
        }
        serde_json::Value::String(s) => Value::string(s.clone()),
        serde_json::Value::Array(arr) => {
            let vals = arr.iter().map(json_to_value).collect::<Vec<_>>();
            crate::make_cell(vals.clone(), 1, vals.len())
                .unwrap_or_else(|_| Value::string("<invalid-array>".to_string()))
        }
        serde_json::Value::Object(map) => {
            let mut s = runmat_value::UnboundedStructValueBuilder::new();
            for (k, v) in map {
                s.fields.insert(k.clone(), json_to_value(v));
            }
            Value::Struct(s.finish())
        }
        serde_json::Value::Null => Value::String(runmat_value::StringElement::Missing),
    }
}
