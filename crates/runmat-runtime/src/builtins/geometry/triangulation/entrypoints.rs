use super::*;

#[runtime_builtin(
    name = "DelaunayTri",
    binding_variant = "default",
    category = "geometry/triangulation",
    summary = "Create a legacy 2-D Delaunay triangulation object.",
    keywords = "DelaunayTri,delaunay,triangulation,geometry,mesh",
    accel = "cpu",
    descriptor(runmat_builtins::DELAUNAY_TRI_DESCRIPTOR),
    extensions(runmat_builtins::DELAUNAY_TRI_CONSTRUCTOR_EXTENSIONS),
    integer_capabilities(runmat_builtins::DELAUNAY_TRI_CONSTRUCTOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DELAUNAY_TRI_DESCRIPTOR,
        &runmat_builtins::DELAUNAY_TRI_ERROR_TOO_MANY_OUTPUTS,
        "DelaunayTri",
    )?;
    ensure_registered();
    ensure_resident_constructor_extensions(&args)?;
    let parsed = parse_constructor_args(args).await?;
    ensure_coordinate_extension(parsed.integer_coordinates)?;
    ensure_topology_extension(parsed.integer_topology)?;
    Ok(Value::Object(delaunay_object(parsed)?))
}

#[runtime_builtin(
    name = "freeBoundary",
    binding_variant = "default",
    category = "geometry/triangulation",
    summary = "Return the boundary edges of a DelaunayTri object.",
    keywords = "freeBoundary,DelaunayTri,triangulation,boundary,mesh",
    accel = "cpu",
    descriptor(runmat_builtins::FREE_BOUNDARY_DESCRIPTOR),
    extensions(runmat_builtins::FREE_BOUNDARY_EXTENSIONS),
    integer_capabilities(runmat_builtins::FREE_BOUNDARY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn free_boundary_builtin(dt: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::FREE_BOUNDARY_DESCRIPTOR,
        &runmat_builtins::FREE_BOUNDARY_ERROR_TOO_MANY_OUTPUTS,
        "freeBoundary",
    )?;
    free_boundary_value(dt)
}

#[runtime_builtin(
    name = "DelaunayTri.freeBoundary",
    binding_variant = "default",
    category = "geometry/triangulation",
    summary = "Return the boundary edges of a DelaunayTri object.",
    keywords = "freeBoundary,DelaunayTri,triangulation,boundary,mesh",
    accel = "cpu",
    descriptor(runmat_builtins::DELAUNAY_TRI_FREE_BOUNDARY_DESCRIPTOR),
    extensions(runmat_builtins::FREE_BOUNDARY_EXTENSIONS),
    integer_capabilities(runmat_builtins::FREE_BOUNDARY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_free_boundary_builtin(dt: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DELAUNAY_TRI_FREE_BOUNDARY_DESCRIPTOR,
        &runmat_builtins::FREE_BOUNDARY_ERROR_TOO_MANY_OUTPUTS,
        "DelaunayTri.freeBoundary",
    )?;
    free_boundary_value(dt)
}

#[runtime_builtin(
    name = "nearestNeighbor",
    binding_variant = "default",
    category = "geometry/triangulation",
    summary = "Find nearest DelaunayTri vertices for query points.",
    keywords = "nearestNeighbor,DelaunayTri,triangulation,geometry",
    accel = "cpu",
    descriptor(runmat_builtins::NEAREST_NEIGHBOR_DESCRIPTOR),
    extensions(runmat_builtins::DELAUNAY_TRI_COORDINATE_EXTENSIONS),
    integer_capabilities(runmat_builtins::NEAREST_NEIGHBOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn nearest_neighbor_builtin(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::NEAREST_NEIGHBOR_DESCRIPTOR,
        &runmat_builtins::NEAREST_NEIGHBOR_ERROR_TOO_MANY_OUTPUTS,
        "nearestNeighbor",
    )?;
    nearest_neighbor_value(dt, rest).await
}

#[runtime_builtin(
    name = "DelaunayTri.nearestNeighbor",
    binding_variant = "default",
    category = "geometry/triangulation",
    summary = "Find nearest DelaunayTri vertices for query points.",
    keywords = "nearestNeighbor,DelaunayTri,triangulation,geometry",
    accel = "cpu",
    descriptor(runmat_builtins::DELAUNAY_TRI_NEAREST_NEIGHBOR_DESCRIPTOR),
    extensions(runmat_builtins::DELAUNAY_TRI_COORDINATE_EXTENSIONS),
    integer_capabilities(runmat_builtins::NEAREST_NEIGHBOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_nearest_neighbor_builtin(
    dt: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DELAUNAY_TRI_NEAREST_NEIGHBOR_DESCRIPTOR,
        &runmat_builtins::NEAREST_NEIGHBOR_ERROR_TOO_MANY_OUTPUTS,
        "DelaunayTri.nearestNeighbor",
    )?;
    nearest_neighbor_value(dt, rest).await
}

#[runtime_builtin(
    name = "pointLocation",
    category = "geometry/triangulation",
    summary = "Locate query points in a DelaunayTri object.",
    keywords = "pointLocation,DelaunayTri,triangulation,barycentric,geometry",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::POINT_LOCATION_DESCRIPTOR),
    extensions(crate::builtins::geometry::triangulation::POINT_LOCATION_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::POINT_LOCATION_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn point_location_builtin(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    point_location_value(dt, rest).await
}

#[runtime_builtin(
    name = "DelaunayTri.pointLocation",
    category = "geometry/triangulation",
    summary = "Locate query points in a DelaunayTri object.",
    keywords = "pointLocation,DelaunayTri,triangulation,barycentric,geometry",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::POINT_LOCATION_DESCRIPTOR),
    extensions(crate::builtins::geometry::triangulation::POINT_LOCATION_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::POINT_LOCATION_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_point_location_builtin(
    dt: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    point_location_value(dt, rest).await
}
