use super::*;

pub(super) fn free_boundary_value(dt: Value) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let integer_topology = object_topology_is_typed_integer(object);
    let triangles = object_triangles(object, None)?;
    ensure_topology_extension(integer_topology)?;
    let edges = boundary_edges(&triangles);
    edges_tensor(&edges)
}

pub(super) async fn nearest_neighbor_value(
    dt: Value,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let object_integer_coordinates = object_coordinates_are_typed_integer(object);
    let points = object_points(object)?;
    ensure_resident_coordinate_values(rest.view())?;
    preflight::query(rest.view(), "nearestNeighbor")?;
    let gathered = gather_values(rest, runtime).await?;
    let query_integer_coordinates = gathered.iter().any(is_typed_integer_numeric_value);
    let queries = parse_query_points(gathered.view(), "nearestNeighbor")?;
    ensure_coordinate_extension(object_integer_coordinates || query_integer_coordinates)?;
    let indices = nearest_neighbor_indices(&points, &queries)
        .into_iter()
        .map(|idx| idx.map_or(f64::NAN, |idx| idx as f64 + 1.0))
        .collect::<Vec<_>>();
    Tensor::new(indices, vec![queries.len(), 1])
        .map(Value::Tensor)
        .map_err(invalid)
}

pub(super) async fn point_location_value(
    dt: Value,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let object_integer_coordinates = object_coordinates_are_typed_integer(object);
    let object_integer_topology = object_topology_is_typed_integer(object);
    let points = object_points(object)?;
    let triangles = object_triangles(object, Some(points.len()))?;
    ensure_resident_coordinate_values(rest.view())?;
    preflight::query(rest.view(), "pointLocation")?;
    let gathered = gather_values(rest, runtime).await?;
    let query_integer_coordinates = gathered.iter().any(is_typed_integer_numeric_value);
    let queries = parse_query_points(gathered.view(), "pointLocation")?;
    ensure_coordinate_extension(object_integer_coordinates || query_integer_coordinates)?;
    ensure_topology_extension(object_integer_topology)?;
    let locations = point_locations(&points, &triangles, &queries);
    let indices = locations
        .iter()
        .map(|(idx, _)| idx.map_or(f64::NAN, |idx| idx as f64 + 1.0))
        .collect::<Vec<_>>();
    let ti = Tensor::new(indices, vec![queries.len(), 1])
        .map(Value::Tensor)
        .map_err(invalid)?;
    if current_requested_outputs() <= 1 {
        return Ok(ti);
    }
    let mut bary = Vec::with_capacity(queries.len() * 3);
    for col in 0..3 {
        for (_, coords) in &locations {
            bary.push(coords[col]);
        }
    }
    let bc = Tensor::new_2d(bary, queries.len(), 3)
        .map(Value::Tensor)
        .map_err(invalid)?;
    Ok(Value::output_list(vec![ti, bc]))
}

pub(super) fn require_delaunay_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(DELAUNAY_TRI_CLASS) => Ok(object),
        _ => Err(invalid("expected DelaunayTri object")),
    }
}

pub(super) fn object_points(object: &ObjectInstance) -> BuiltinResult<Vec<[f64; 2]>> {
    let value = object
        .properties()
        .get(POINTS_PROPERTY)
        .or_else(|| object.properties().get(LEGACY_POINTS_PROPERTY))
        .ok_or_else(|| invalid("DelaunayTri object is missing point coordinates"))?;
    matrix_points(value, DELAUNAY_TRI_CLASS.display_name())
}

pub(super) fn object_triangles(
    object: &ObjectInstance,
    point_count: Option<usize>,
) -> BuiltinResult<Vec<[usize; 3]>> {
    let value = object
        .properties()
        .get(CONNECTIVITY_PROPERTY)
        .or_else(|| object.properties().get(LEGACY_CONNECTIVITY_PROPERTY))
        .ok_or_else(|| invalid("DelaunayTri object is missing connectivity"))?;
    connectivity_from_value(value, point_count)
}

pub(super) fn parse_query_points(
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<Vec<[f64; 2]>> {
    match args.len() {
        1 => matrix_points(&args[0], builtin),
        2 => vector_points(&args[0], &args[1], builtin),
        _ => Err(invalid(format!(
            "{builtin}: expected query matrix or x,y vectors"
        ))),
    }
}
