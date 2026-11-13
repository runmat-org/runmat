use super::*;

pub(super) struct ParsedDelaunay {
    pub(super) points: Vec<[f64; 2]>,
    pub(super) constraints: Vec<[usize; 2]>,
    pub(super) integer_coordinates: bool,
    pub(super) integer_topology: bool,
}

pub(super) async fn parse_constructor_args(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<ParsedDelaunay> {
    preflight::constructor(args.view())?;
    let gathered = gather_values(args, runtime).await?;
    match gathered.len() {
        0 => Ok(ParsedDelaunay {
            points: Vec::new(),
            constraints: Vec::new(),
            integer_coordinates: false,
            integer_topology: false,
        }),
        1 => {
            let points = &gathered[0];
            Ok(ParsedDelaunay {
                points: matrix_points(points, BUILTIN_NAME)?,
                constraints: Vec::new(),
                integer_coordinates: is_typed_integer_numeric_value(points),
                integer_topology: false,
            })
        }
        2 if constraint_form(&gathered[0], &gathered[1]) => {
            let first = &gathered[0];
            let second = &gathered[1];
            let constraints = constraints_from_value(second)?;
            if !constraints.is_empty() {
                return Err(unsupported(
                    "constrained DelaunayTri triangulation is not yet supported",
                ));
            }
            Ok(ParsedDelaunay {
                points: matrix_points(first, BUILTIN_NAME)?,
                constraints,
                integer_coordinates: is_typed_integer_numeric_value(first),
                integer_topology: is_typed_integer_numeric_value(second),
            })
        }
        2 => {
            let x = &gathered[0];
            let y = &gathered[1];
            Ok(ParsedDelaunay {
                points: vector_points(x, y, BUILTIN_NAME)?,
                constraints: Vec::new(),
                integer_coordinates: is_typed_integer_numeric_value(x)
                    || is_typed_integer_numeric_value(y),
                integer_topology: false,
            })
        }
        3 => {
            let x = &gathered[0];
            let y = &gathered[1];
            let third = &gathered[2];
            if looks_like_constraints(third) {
                let constraints = constraints_from_value(third)?;
                if !constraints.is_empty() {
                    return Err(unsupported(
                        "constrained DelaunayTri triangulation is not yet supported",
                    ));
                }
                Ok(ParsedDelaunay {
                    points: vector_points(x, y, BUILTIN_NAME)?,
                    constraints,
                    integer_coordinates: is_typed_integer_numeric_value(x)
                        || is_typed_integer_numeric_value(y),
                    integer_topology: is_typed_integer_numeric_value(third),
                })
            } else {
                Err(unsupported(
                    "3-D DelaunayTri triangulation is not yet supported",
                ))
            }
        }
        _ => Err(invalid(
            "expected DelaunayTri(), DelaunayTri(X), DelaunayTri(x,y), or DelaunayTri(X,C)",
        )),
    }
}

pub(super) async fn gather_values(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    crate::dispatcher::gather_builtin_arguments_async(args, runtime).await
}

pub(super) fn ensure_resident_constructor_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    match args.len() {
        1 => ensure_resident_coordinate_values(args),
        2 if constraint_form(&args[0], &args[1]) => {
            ensure_resident_coordinate_values(args.range(..1).expect("first argument"))?;
            ensure_resident_topology_values(args.range(1..).expect("second argument"))
        }
        2 => ensure_resident_coordinate_values(args),
        3 => {
            ensure_resident_coordinate_values(args.range(..2).expect("coordinate arguments"))?;
            ensure_resident_topology_values(args.range(2..).expect("topology argument"))
        }
        _ => Ok(()),
    }
}

pub(super) fn ensure_resident_coordinate_values(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if values.iter().any(is_resident_typed_integer_value) {
        ensure_coordinate_extension(true)?;
    }
    Ok(())
}

pub(super) fn ensure_resident_topology_values(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if values.iter().any(is_resident_typed_integer_value) {
        ensure_topology_extension(true)?;
    }
    Ok(())
}

pub(super) fn ensure_coordinate_extension(enabled: bool) -> BuiltinResult<()> {
    if enabled {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_COORDINATES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn ensure_topology_extension(enabled: bool) -> BuiltinResult<()> {
    if enabled {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_TOPOLOGY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn is_typed_integer_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || is_resident_typed_integer_value(value)
}

pub(super) fn is_resident_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn object_coordinates_are_typed_integer(object: &ObjectInstance) -> bool {
    object
        .properties()
        .get(POINTS_PROPERTY)
        .or_else(|| object.properties().get(LEGACY_POINTS_PROPERTY))
        .is_some_and(is_typed_integer_numeric_value)
}

pub(super) fn object_topology_is_typed_integer(object: &ObjectInstance) -> bool {
    object
        .properties()
        .get(CONNECTIVITY_PROPERTY)
        .or_else(|| object.properties().get(LEGACY_CONNECTIVITY_PROPERTY))
        .is_some_and(is_typed_integer_numeric_value)
}

pub(super) fn delaunay_object(parsed: ParsedDelaunay) -> BuiltinResult<ObjectInstance> {
    let mesh = delaunay_2d(&parsed.points).map_err(|err| invalid(err.to_string()))?;
    let points = points_tensor(&parsed.points)?;
    let triangles = triangles_tensor(&mesh.triangles)?;
    let constraints = constraints_tensor(&parsed.constraints)?;

    let mut object = runmat_value::UnboundedObjectBuilder::with_property_capacity(
        DELAUNAY_TRI_CLASS.to_string(),
        6,
    );
    object.insert(LEGACY_POINTS_PROPERTY.to_string(), points.clone());
    object.insert(POINTS_PROPERTY.to_string(), points);
    object.insert(LEGACY_CONNECTIVITY_PROPERTY.to_string(), triangles.clone());
    object.insert(CONNECTIVITY_PROPERTY.to_string(), triangles);
    object.insert(CONSTRAINTS_PROPERTY.to_string(), constraints);
    object.insert(DIMENSION_PROPERTY.to_string(), Value::Num(2.0));
    Ok(object.finish())
}
