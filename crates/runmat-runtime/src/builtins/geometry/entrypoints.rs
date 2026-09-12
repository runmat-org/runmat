use super::*;

#[runtime_builtin(
    name = "geometry.load",
    category = "geometry",
    summary = "Load a geometry file into a structured geometry asset.",
    keywords = "geometry,load,cad,mesh,stl,step,obj",
    descriptor(crate::builtins::geometry::GEOMETRY_LOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_load_builtin(path: String) -> BuiltinResult<Value> {
    let bytes = read_file(GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_LOAD_NAME,
        &GEOMETRY_LOAD_ERROR_OPERATION,
        &GEOMETRY_LOAD_ERROR_INTERNAL,
        crate::geometry::geometry_load_op(&path, &bytes, OperationContext::new(None, None)),
        Some(GEOMETRY_ASSET_CLASS),
        Some(GEOMETRY_ASSET_JSON_PROPERTY),
    )
}

#[runtime_builtin(
    name = "geometry.inspect",
    category = "geometry",
    summary = "Inspect a geometry file without importing the full asset.",
    keywords = "geometry,inspect,cad,mesh,stl,step,obj",
    descriptor(crate::builtins::geometry::GEOMETRY_INSPECT_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_inspect_builtin(path: String) -> BuiltinResult<Value> {
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
    category = "geometry",
    summary = "List regions imported into a geometry asset.",
    keywords = "geometry,regions,cad,selectors,fea",
    descriptor(crate::builtins::geometry::GEOMETRY_LIST_REGIONS_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::GEOMETRY_LIST_REGIONS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_list_regions_builtin(asset: Value) -> BuiltinResult<Value> {
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
    category = "geometry",
    summary = "Return renderable surface mesh topology for a geometry asset.",
    keywords = "geometry,mesh,vertices,triangles,faces,patch,fea",
    descriptor(crate::builtins::geometry::GEOMETRY_MESHES_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::GEOMETRY_MESHES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_meshes_builtin(asset: Value) -> BuiltinResult<Value> {
    let asset = geometry_asset_from_value_with_builtin(&asset, GEOMETRY_MESHES_NAME)?;
    geometry_meshes_value(&asset)
}
