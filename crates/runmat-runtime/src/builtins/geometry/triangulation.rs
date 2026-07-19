//! MATLAB-compatible triangulation objects and helpers.

use std::collections::HashMap;
use std::sync::OnceLock;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, NumericDType, ObjectInstance, PropertyDef, Tensor, Value,
};
use runmat_geometry_ops::{boundary_edges, delaunay_2d, nearest_neighbor_indices, point_locations};
use runmat_macros::runtime_builtin;

use crate::{
    build_runtime_error, current_requested_outputs, gather_if_needed_async, BuiltinResult,
};

pub const DELAUNAY_TRI_CLASS: &str = "DelaunayTri";
const BUILTIN_NAME: &str = "DelaunayTri";
const POINTS_PROPERTY: &str = "Points";
const CONNECTIVITY_PROPERTY: &str = "ConnectivityList";
const LEGACY_POINTS_PROPERTY: &str = "X";
const LEGACY_CONNECTIVITY_PROPERTY: &str = "Triangulation";
const CONSTRAINTS_PROPERTY: &str = "Constraints";
const DIMENSION_PROPERTY: &str = "Dimension";

const OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy DelaunayTri triangulation object.",
}];

const INPUT_VARARGIN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Point matrix, coordinate vectors, and optional constraints.",
}];

const DELAUNAY_TRI_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri()",
        inputs: &[],
        outputs: &OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(X)",
        inputs: &INPUT_VARARGIN,
        outputs: &OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(x, y)",
        inputs: &INPUT_VARARGIN,
        outputs: &OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(X, C)",
        inputs: &INPUT_VARARGIN,
        outputs: &OUTPUT_OBJECT,
    },
];

const OUTPUT_TENSOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric result.",
}];

const OUTPUT_POINT_LOCATION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ti",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Containing triangle indices, or NaN outside the triangulation.",
    },
    BuiltinParamDescriptor {
        name: "bc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Barycentric coordinates for each query point.",
    },
];

const INPUT_OBJECT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "DelaunayTri object.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Query point matrix or coordinate vectors.",
    },
];

const HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = helper(dt, varargin)",
    inputs: &INPUT_OBJECT_REST,
    outputs: &OUTPUT_TENSOR,
}];

const POINT_LOCATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[ti, bc] = pointLocation(dt, q)",
    inputs: &INPUT_OBJECT_REST,
    outputs: &OUTPUT_POINT_LOCATION,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELAUNAY_TRI.INVALID_ARGUMENT",
    identifier: Some("RunMat:DelaunayTri:InvalidArgument"),
    when: "Constructor, object, constraint, or query inputs are malformed.",
    message: "DelaunayTri: invalid argument",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELAUNAY_TRI.UNSUPPORTED",
    identifier: Some("RunMat:DelaunayTri:Unsupported"),
    when: "The legacy class is used with constrained or 3-D triangulation forms that RunMat does not implement yet.",
    message: "DelaunayTri: unsupported form",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_UNSUPPORTED];

pub const DELAUNAY_TRI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELAUNAY_TRI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HELPER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const POINT_LOCATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_LOCATION_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "DelaunayTri",
    category = "geometry/triangulation",
    summary = "Create a legacy 2-D Delaunay triangulation object.",
    keywords = "DelaunayTri,delaunay,triangulation,geometry,mesh",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::DELAUNAY_TRI_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_registered();
    let parsed = parse_constructor_args(args).await?;
    Ok(Value::Object(delaunay_object(parsed)?))
}

#[runtime_builtin(
    name = "freeBoundary",
    category = "geometry/triangulation",
    summary = "Return the boundary edges of a DelaunayTri object.",
    keywords = "freeBoundary,DelaunayTri,triangulation,boundary,mesh",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn free_boundary_builtin(dt: Value) -> BuiltinResult<Value> {
    free_boundary_value(dt)
}

#[runtime_builtin(
    name = "DelaunayTri.freeBoundary",
    category = "geometry/triangulation",
    summary = "Return the boundary edges of a DelaunayTri object.",
    keywords = "freeBoundary,DelaunayTri,triangulation,boundary,mesh",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_free_boundary_builtin(dt: Value) -> BuiltinResult<Value> {
    free_boundary_value(dt)
}

#[runtime_builtin(
    name = "nearestNeighbor",
    category = "geometry/triangulation",
    summary = "Find nearest DelaunayTri vertices for query points.",
    keywords = "nearestNeighbor,DelaunayTri,triangulation,geometry",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn nearest_neighbor_builtin(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    nearest_neighbor_value(dt, rest).await
}

#[runtime_builtin(
    name = "DelaunayTri.nearestNeighbor",
    category = "geometry/triangulation",
    summary = "Find nearest DelaunayTri vertices for query points.",
    keywords = "nearestNeighbor,DelaunayTri,triangulation,geometry",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_nearest_neighbor_builtin(
    dt: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    nearest_neighbor_value(dt, rest).await
}

#[runtime_builtin(
    name = "pointLocation",
    category = "geometry/triangulation",
    summary = "Locate query points in a DelaunayTri object.",
    keywords = "pointLocation,DelaunayTri,triangulation,barycentric,geometry",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::POINT_LOCATION_DESCRIPTOR),
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
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_point_location_builtin(
    dt: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    point_location_value(dt, rest).await
}

pub(crate) fn register_delaunaytri_class() {
    ensure_registered();
}

fn ensure_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        let mut properties = HashMap::new();
        for name in [
            LEGACY_POINTS_PROPERTY,
            LEGACY_CONNECTIVITY_PROPERTY,
            CONSTRAINTS_PROPERTY,
            POINTS_PROPERTY,
            CONNECTIVITY_PROPERTY,
            DIMENSION_PROPERTY,
        ] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for name in ["freeBoundary", "nearestNeighbor", "pointLocation"] {
            methods.insert(
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{DELAUNAY_TRI_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: DELAUNAY_TRI_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

struct ParsedDelaunay {
    points: Vec<[f64; 2]>,
    constraints: Vec<[usize; 2]>,
}

async fn parse_constructor_args(args: Vec<Value>) -> BuiltinResult<ParsedDelaunay> {
    let gathered = gather_values(args).await?;
    match gathered.as_slice() {
        [] => Ok(ParsedDelaunay {
            points: Vec::new(),
            constraints: Vec::new(),
        }),
        [points] => Ok(ParsedDelaunay {
            points: matrix_points(points, BUILTIN_NAME)?,
            constraints: Vec::new(),
        }),
        [first, second] if looks_like_point_matrix(first) && !is_vector_value(first) => {
            let constraints = constraints_from_value(second)?;
            if !constraints.is_empty() {
                return Err(unsupported(
                    "constrained DelaunayTri triangulation is not yet supported",
                ));
            }
            Ok(ParsedDelaunay {
                points: matrix_points(first, BUILTIN_NAME)?,
                constraints,
            })
        }
        [x, y] => Ok(ParsedDelaunay {
            points: vector_points(x, y, BUILTIN_NAME)?,
            constraints: Vec::new(),
        }),
        [x, y, third] => {
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

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

fn delaunay_object(parsed: ParsedDelaunay) -> BuiltinResult<ObjectInstance> {
    let mesh = delaunay_2d(&parsed.points).map_err(|err| invalid(err.to_string()))?;
    let points = points_tensor(&parsed.points)?;
    let triangles = triangles_tensor(&mesh.triangles)?;
    let constraints = constraints_tensor(&parsed.constraints)?;

    let mut object = ObjectInstance::new(DELAUNAY_TRI_CLASS.to_string());
    object
        .properties
        .insert(LEGACY_POINTS_PROPERTY.to_string(), points.clone());
    object
        .properties
        .insert(POINTS_PROPERTY.to_string(), points);
    object
        .properties
        .insert(LEGACY_CONNECTIVITY_PROPERTY.to_string(), triangles.clone());
    object
        .properties
        .insert(CONNECTIVITY_PROPERTY.to_string(), triangles);
    object
        .properties
        .insert(CONSTRAINTS_PROPERTY.to_string(), constraints);
    object
        .properties
        .insert(DIMENSION_PROPERTY.to_string(), Value::Num(2.0));
    Ok(object)
}

fn free_boundary_value(dt: Value) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let triangles = object_triangles(object, None)?;
    let edges = boundary_edges(&triangles);
    edges_tensor(&edges)
}

async fn nearest_neighbor_value(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let points = object_points(object)?;
    let queries = parse_query_points(gather_values(rest).await?, "nearestNeighbor")?;
    let indices = nearest_neighbor_indices(&points, &queries)
        .into_iter()
        .map(|idx| idx.map_or(f64::NAN, |idx| idx as f64 + 1.0))
        .collect::<Vec<_>>();
    Tensor::new(indices, vec![queries.len(), 1])
        .map(Value::Tensor)
        .map_err(invalid)
}

async fn point_location_value(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let points = object_points(object)?;
    let triangles = object_triangles(object, Some(points.len()))?;
    let queries = parse_query_points(gather_values(rest).await?, "pointLocation")?;
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
    Ok(Value::OutputList(vec![ti, bc]))
}

fn require_delaunay_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(DELAUNAY_TRI_CLASS) => Ok(object),
        _ => Err(invalid("expected DelaunayTri object")),
    }
}

fn object_points(object: &ObjectInstance) -> BuiltinResult<Vec<[f64; 2]>> {
    let value = object
        .properties
        .get(POINTS_PROPERTY)
        .or_else(|| object.properties.get(LEGACY_POINTS_PROPERTY))
        .ok_or_else(|| invalid("DelaunayTri object is missing point coordinates"))?;
    matrix_points(value, DELAUNAY_TRI_CLASS)
}

fn object_triangles(
    object: &ObjectInstance,
    point_count: Option<usize>,
) -> BuiltinResult<Vec<[usize; 3]>> {
    let value = object
        .properties
        .get(CONNECTIVITY_PROPERTY)
        .or_else(|| object.properties.get(LEGACY_CONNECTIVITY_PROPERTY))
        .ok_or_else(|| invalid("DelaunayTri object is missing connectivity"))?;
    connectivity_from_value(value, point_count)
}

fn parse_query_points(args: Vec<Value>, builtin: &'static str) -> BuiltinResult<Vec<[f64; 2]>> {
    match args.as_slice() {
        [q] => matrix_points(q, builtin),
        [x, y] => vector_points(x, y, builtin),
        _ => Err(invalid(format!(
            "{builtin}: expected query matrix or x,y vectors"
        ))),
    }
}

fn matrix_points(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<[f64; 2]>> {
    let tensor = numeric_tensor(value, builtin)?;
    if tensor.cols == 3 {
        return Err(unsupported(
            "3-D DelaunayTri triangulation is not yet supported",
        ));
    }
    if tensor.cols < 2 {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    let mut points = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let x = tensor.data[row];
        let y = tensor.data[row + tensor.rows];
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid(format!("{builtin}: points must be finite")));
        }
        points.push([x, y]);
    }
    Ok(points)
}

fn vector_points(x: &Value, y: &Value, builtin: &'static str) -> BuiltinResult<Vec<[f64; 2]>> {
    let x = numeric_tensor(x, builtin)?;
    let y = numeric_tensor(y, builtin)?;
    if !is_vector(&x) || !is_vector(&y) || x.data.len() != y.data.len() {
        return Err(invalid(format!(
            "{builtin}: x and y must be vectors with the same length"
        )));
    }
    x.data
        .iter()
        .zip(&y.data)
        .map(|(x, y)| {
            if !x.is_finite() || !y.is_finite() {
                Err(invalid(format!("{builtin}: points must be finite")))
            } else {
                Ok([*x, *y])
            }
        })
        .collect()
}

fn constraints_from_value(value: &Value) -> BuiltinResult<Vec<[usize; 2]>> {
    let tensor = numeric_tensor(value, BUILTIN_NAME)?;
    if tensor.data.is_empty() || tensor.rows == 0 {
        return Ok(Vec::new());
    }
    if tensor.cols != 2 {
        return Err(invalid("DelaunayTri: constraints must be an M-by-2 matrix"));
    }
    let mut constraints = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        constraints.push([
            positive_index(tensor.data[row], usize::MAX)?,
            positive_index(tensor.data[row + tensor.rows], usize::MAX)?,
        ]);
    }
    Ok(constraints)
}

fn connectivity_from_value(
    value: &Value,
    point_count: Option<usize>,
) -> BuiltinResult<Vec<[usize; 3]>> {
    let tensor = numeric_tensor(value, DELAUNAY_TRI_CLASS)?;
    if tensor.cols != 3 {
        return Err(invalid("DelaunayTri connectivity must be an M-by-3 matrix"));
    }
    let mut triangles = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        triangles.push([
            positive_index(tensor.data[row], usize::MAX)? - 1,
            positive_index(tensor.data[row + tensor.rows], usize::MAX)? - 1,
            positive_index(tensor.data[row + 2 * tensor.rows], usize::MAX)? - 1,
        ]);
    }
    if let Some(point_count) = point_count {
        for tri in &triangles {
            if tri.iter().any(|idx| *idx >= point_count) {
                return Err(invalid(
                    "DelaunayTri connectivity references a missing point",
                ));
            }
        }
    }
    Ok(triangles)
}

fn numeric_tensor(value: &Value, builtin: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::Num(value) => Tensor::new(vec![*value], vec![1, 1]).map_err(invalid),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1]).map_err(invalid),
        other => Err(invalid(format!(
            "{builtin}: expected numeric input, got {other:?}"
        ))),
    }
}

fn looks_like_point_matrix(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols >= 2)
}

fn looks_like_constraints(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols == 2)
}

fn is_vector_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if is_vector(tensor))
        || matches!(value, Value::Num(_) | Value::Int(_))
}

fn is_vector(tensor: &Tensor) -> bool {
    tensor.rows == 1 || tensor.cols == 1
}

fn points_tensor(points: &[[f64; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(points.len() * 2);
    for col in 0..2 {
        for point in points {
            data.push(point[col]);
        }
    }
    Tensor::new_2d(data, points.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

fn triangles_tensor(triangles: &[[usize; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(triangles.len() * 3);
    for col in 0..3 {
        for tri in triangles {
            data.push(tri[col] as f64 + 1.0);
        }
    }
    let mut tensor = Tensor::new_2d(data, triangles.len(), 3).map_err(invalid)?;
    tensor.dtype = NumericDType::F64;
    Ok(Value::Tensor(tensor))
}

fn constraints_tensor(constraints: &[[usize; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(constraints.len() * 2);
    for col in 0..2 {
        for constraint in constraints {
            data.push(constraint[col] as f64 + 1.0);
        }
    }
    Tensor::new_2d(data, constraints.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

fn edges_tensor(edges: &[[usize; 2]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(edges.len() * 2);
    for col in 0..2 {
        for edge in edges {
            data.push(edge[col] as f64 + 1.0);
        }
    }
    Tensor::new_2d(data, edges.len(), 2)
        .map(Value::Tensor)
        .map_err(invalid)
}

fn positive_index(value: f64, max: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid("indices must be positive finite integers"));
    }
    let idx = value as usize;
    if idx == 0 || idx > max {
        return Err(invalid("index is out of range"));
    }
    Ok(idx)
}

fn invalid(detail: impl AsRef<str>) -> crate::RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported(detail: impl AsRef<str>) -> crate::RuntimeError {
    error(&ERROR_UNSUPPORTED, detail)
}

fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(descriptor.identifier.unwrap_or("RunMat:DelaunayTri:Error"))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor {
            data: data.to_vec(),
            rows,
            cols,
            shape: vec![rows, cols],
            dtype: NumericDType::F64,
        })
    }

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object")
        };
        object
    }

    #[test]
    fn delaunaytri_builds_legacy_and_modern_properties() {
        let dt = object(
            block_on(delaunay_tri_builtin(vec![tensor(
                &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                4,
                2,
            )]))
            .unwrap(),
        );
        assert_eq!(dt.class_name, DELAUNAY_TRI_CLASS);
        let Value::Tensor(points) = dt.properties.get("X").unwrap() else {
            panic!("expected points")
        };
        assert_eq!((points.rows, points.cols), (4, 2));
        let Value::Tensor(tri) = dt.properties.get("Triangulation").unwrap() else {
            panic!("expected triangulation")
        };
        assert_eq!(tri.cols, 3);
        assert_eq!(tri.rows, 2);
        assert!(dt.properties.contains_key("ConnectivityList"));
        assert!(dt.properties.contains_key("Points"));
    }

    #[test]
    fn delaunaytri_accepts_coordinate_vectors() {
        let dt = object(
            block_on(delaunay_tri_builtin(vec![
                tensor(&[0.0, 1.0, 0.0], 1, 3),
                tensor(&[0.0, 0.0, 1.0], 1, 3),
            ]))
            .unwrap(),
        );
        let Value::Tensor(points) = dt.properties.get("Points").unwrap() else {
            panic!("expected points")
        };
        assert_eq!(points.rows, 3);
        assert_eq!(points.cols, 2);
    }

    #[test]
    fn delaunaytri_empty_constructor_returns_empty_topology() {
        let dt = object(block_on(delaunay_tri_builtin(vec![])).unwrap());
        let Value::Tensor(points) = dt.properties.get("X").unwrap() else {
            panic!("expected points")
        };
        let Value::Tensor(tri) = dt.properties.get("Triangulation").unwrap() else {
            panic!("expected triangulation")
        };
        assert_eq!((points.rows, points.cols), (0, 2));
        assert_eq!((tri.rows, tri.cols), (0, 3));
    }

    #[test]
    fn delaunaytri_rejects_3d_and_nonempty_constraints_deterministically() {
        let err = block_on(delaunay_tri_builtin(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            3,
            3,
        )]))
        .expect_err("3-D should be explicit");
        assert_eq!(err.identifier(), ERROR_UNSUPPORTED.identifier);

        let err = block_on(delaunay_tri_builtin(vec![
            tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
            tensor(&[1.0, 2.0], 1, 2),
        ]))
        .expect_err("constraints should be explicit");
        assert_eq!(err.identifier(), ERROR_UNSUPPORTED.identifier);
    }

    #[test]
    fn delaunaytri_query_helpers_return_expected_shapes() {
        let dt = block_on(delaunay_tri_builtin(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
        )]))
        .unwrap();
        let Value::Tensor(edges) = block_on(free_boundary_builtin(dt.clone())).unwrap() else {
            panic!("expected boundary tensor")
        };
        assert_eq!((edges.rows, edges.cols), (3, 2));

        let Value::Tensor(nn) = block_on(nearest_neighbor_builtin(
            dt.clone(),
            vec![tensor(&[0.1, 0.8], 1, 2)],
        ))
        .unwrap() else {
            panic!("expected nearest tensor")
        };
        assert_eq!(nn.data, vec![3.0]);

        let location = block_on(point_location_builtin(
            dt,
            vec![tensor(&[0.25, 0.25], 1, 2)],
        ))
        .unwrap();
        let Value::Tensor(ti) = location else {
            panic!("expected location tensor")
        };
        assert_eq!(ti.rows, 1);
        assert_eq!(ti.cols, 1);
        assert_eq!(ti.data, vec![1.0]);
    }

    #[test]
    fn point_location_rejects_inconsistent_object_connectivity() {
        let mut object = object(
            block_on(delaunay_tri_builtin(vec![tensor(
                &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                3,
                2,
            )]))
            .unwrap(),
        );
        object.properties.insert(
            CONNECTIVITY_PROPERTY.to_string(),
            tensor(&[1.0, 2.0, 9.0], 1, 3),
        );
        let err = block_on(point_location_builtin(
            Value::Object(object),
            vec![tensor(&[0.25, 0.25], 1, 2)],
        ))
        .expect_err("bad object connectivity should be rejected");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }
}
