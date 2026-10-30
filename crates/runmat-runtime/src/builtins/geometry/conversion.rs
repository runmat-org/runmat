use super::*;

/// Preserve the previously admitted scalar string/string-array and character
/// row forms without the broad `String::try_from(Value)` numeric coercion.
/// Admission must precede any filesystem provider access.
pub(super) fn geometry_path_from_value(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(path)) => Ok(path.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        Value::CharArray(array) if array.rows == 1 && array.shape.len() <= 2 => Ok(
            runmat_value::decode_utf16_code_units(array.data()).map_err(|source| {
                let mut error = builtin_error(builtin, error, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => Err(builtin_error(
            builtin,
            error,
            "expected a string scalar or character row path",
        )),
    }
}

pub(super) async fn read_file(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    path: &str,
) -> BuiltinResult<Vec<u8>> {
    runmat_filesystem::read_async(path)
        .await
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

pub(super) fn geometry_asset_from_value(value: &Value) -> BuiltinResult<GeometryAsset> {
    geometry_asset_from_value_with_builtin(value, GEOMETRY_LIST_REGIONS_NAME)
}

pub(crate) fn geometry_asset_from_value_with_builtin(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeometryAsset> {
    geometry_asset_from_value_with_error(value, builtin, &GEOMETRY_INSPECT_ERROR_INTERNAL)
}

/// Decode a `geometry.Asset` argument, reporting a non-asset value under the
/// caller's own error descriptor.
pub(crate) fn geometry_asset_from_value_with_error(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            error,
            format!("{builtin}: expected geometry.Asset"),
        ));
    };
    if !object.class_name().is(GEOMETRY_ASSET_CLASS) {
        return Err(builtin_error(
            builtin,
            error,
            format!(
                "{builtin}: expected {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name()
            ),
        ));
    }
    object_json_property(
        builtin,
        object,
        GEOMETRY_ASSET_JSON_PROPERTY,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
    )
}

pub(super) fn geometry_meshes_value(asset: &GeometryAsset) -> BuiltinResult<Value> {
    let values = asset
        .surface_meshes
        .iter()
        .map(|surface| {
            let mut mesh = runmat_value::UnboundedStructValueBuilder::new();
            mesh.insert("mesh_id", Value::string(surface.mesh_id.clone()));
            mesh.insert("vertices", vertices_tensor(&surface.vertices)?);
            mesh.insert("triangles", triangles_tensor(&surface.triangles)?);
            mesh.insert("faces", triangles_tensor(&surface.triangles)?);
            mesh.insert(
                "region_mappings",
                region_mappings_value(asset, &surface.mesh_id)?,
            );
            Ok(Value::Struct(mesh.finish()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    crate::make_cell_with_shape(values, vec![1, asset.surface_meshes.len()])
        .map_err(|err| builtin_error(GEOMETRY_MESHES_NAME, &GEOMETRY_INSPECT_ERROR_INTERNAL, err))
}

pub(super) fn vertices_tensor(vertices: &[[f64; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(vertices.len() * 3);
    for col in 0..3 {
        for vertex in vertices {
            data.push(vertex[col]);
        }
    }
    Tensor::new_2d(data, vertices.len(), 3)
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                GEOMETRY_MESHES_NAME,
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("geometry.meshes: failed to build vertices tensor: {err}"),
            )
        })
}

pub(super) fn triangles_tensor(triangles: &[[u32; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(triangles.len() * 3);
    for col in 0..3 {
        for triangle in triangles {
            data.push(f64::from(triangle[col]) + 1.0);
        }
    }
    Tensor::new_2d(data, triangles.len(), 3)
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                GEOMETRY_MESHES_NAME,
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("geometry.meshes: failed to build triangle tensor: {err}"),
            )
        })
}

pub(super) fn region_mappings_value(asset: &GeometryAsset, mesh_id: &str) -> BuiltinResult<Value> {
    let values = asset
        .region_entity_mappings
        .iter()
        .filter(|mapping| mapping.mesh_id == mesh_id)
        .map(|mapping| {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            value.insert("region_id", Value::string(mapping.region_id.clone()));
            value.insert(
                "entity_kind",
                Value::string(format!("{:?}", mapping.entity_kind).to_ascii_lowercase()),
            );
            let mut ranges = Vec::with_capacity(mapping.ranges.len() * 2);
            for col in 0..2 {
                for range in &mapping.ranges {
                    ranges.push(if col == 0 {
                        range.start as f64 + 1.0
                    } else {
                        range.count as f64
                    });
                }
            }
            let range_tensor = Tensor::new_2d(ranges, mapping.ranges.len(), 2)
                .map(Value::Tensor)
                .map_err(|err| {
                    builtin_error(
                        GEOMETRY_MESHES_NAME,
                        &GEOMETRY_INSPECT_ERROR_INTERNAL,
                        format!("geometry.meshes: failed to build range tensor: {err}"),
                    )
                })?;
            value.insert("ranges", range_tensor);
            Ok(Value::Struct(value.finish()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let cols = values.len();
    crate::make_cell_with_shape(values, vec![1, cols])
        .map_err(|err| builtin_error(GEOMETRY_MESHES_NAME, &GEOMETRY_INSPECT_ERROR_INTERNAL, err))
}

pub(super) fn object_json_property<T: DeserializeOwned>(
    builtin: &'static str,
    object: &ObjectInstance,
    property: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    let Some(Value::String(runmat_value::StringElement::Text(json))) =
        object.properties().get(property)
    else {
        return Err(build_runtime_error(format!(
            "{} is missing required runtime payload property `{property}`",
            object.class_name()
        ))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:geometry:Internal"))
        .build());
    };
    serde_json::from_str(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

pub(super) fn operation_result_to_value<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    class_identity: Option<runmat_types::StaticClassIdentity>,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    match class_identity {
        Some(class_identity) => serializable_to_object(
            builtin,
            internal_error_descriptor,
            class_identity,
            &envelope.data,
            hidden_json_property,
        ),
        None => serializable_to_value(builtin, internal_error_descriptor, &envelope.data),
    }
}

pub(super) fn serializable_to_value<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &T,
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    value_from_json(&json)
}

pub(super) fn serializable_to_object<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_identity: runmat_types::StaticClassIdentity,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    ensure_geometry_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let converted = value_from_json(&json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    let mut object = runmat_value::UnboundedObjectBuilder::new(class_identity);
    if let Value::Struct(fields) = converted {
        for (name, value) in fields.into_fields_unbounded() {
            object.insert(name, value);
        }
    } else {
        object.insert("value".to_string(), converted);
    }
    if let Some(property) = hidden_json_property {
        object.insert(property.to_string(), Value::string(json.to_string()));
    }
    Ok(Value::Object(object.finish()))
}
