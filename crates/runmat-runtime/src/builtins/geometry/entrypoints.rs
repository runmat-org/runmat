use super::*;

#[runtime_builtin(
    name = "geometry.load",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_load_builtin(
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GEOMETRY_LOAD_DESCRIPTOR,
        &GEOMETRY_LOAD_ERROR_OUTPUTS,
        GEOMETRY_LOAD_NAME,
    )?;
    let path = geometry_path_from_value(&path, GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_INPUT)?;
    let import_options = super::display::parse_load_options(rest.view())?;
    let bytes = read_file(GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_LOAD_NAME,
        &GEOMETRY_LOAD_ERROR_OPERATION,
        &GEOMETRY_LOAD_ERROR_INTERNAL,
        crate::geometry::geometry_load_with_options_op(
            &path,
            &bytes,
            import_options,
            OperationContext::new(None, None),
        ),
        Some(GEOMETRY_ASSET_CLASS),
        Some(GEOMETRY_ASSET_JSON_PROPERTY),
    )
}

#[runtime_builtin(
    name = "geometry.inspect",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_inspect_builtin(path: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GEOMETRY_INSPECT_DESCRIPTOR,
        &GEOMETRY_INSPECT_ERROR_OUTPUTS,
        GEOMETRY_INSPECT_NAME,
    )?;
    let path =
        geometry_path_from_value(&path, GEOMETRY_INSPECT_NAME, &GEOMETRY_INSPECT_ERROR_INPUT)?;
    let bytes = read_file(GEOMETRY_INSPECT_NAME, &GEOMETRY_INSPECT_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_INSPECT_NAME,
        &GEOMETRY_INSPECT_ERROR_OPERATION,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
        crate::geometry::geometry_inspect_op(&path, &bytes, OperationContext::new(None, None)),
        Some(GEOMETRY_INSPECT_RESULT_CLASS),
        None,
    )
}

#[runtime_builtin(
    name = "geometry.listRegions",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_list_regions_builtin(asset: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GEOMETRY_LIST_REGIONS_DESCRIPTOR,
        &GEOMETRY_LIST_REGIONS_ERROR_OUTPUTS,
        GEOMETRY_LIST_REGIONS_NAME,
    )?;
    let asset = geometry_asset_from_value(&asset)?;
    operation_result_to_value(
        GEOMETRY_LIST_REGIONS_NAME,
        &GEOMETRY_INSPECT_ERROR_OPERATION,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
        crate::geometry::geometry_list_regions_op(&asset, OperationContext::new(None, None)),
        None,
        None,
    )
}

#[runtime_builtin(
    name = "geometry.meshes",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_meshes_builtin(asset: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GEOMETRY_MESHES_DESCRIPTOR,
        &GEOMETRY_MESHES_ERROR_OUTPUTS,
        GEOMETRY_MESHES_NAME,
    )?;
    let asset = geometry_asset_from_value_with_builtin(&asset, GEOMETRY_MESHES_NAME)?;
    geometry_meshes_value(&asset)
}
