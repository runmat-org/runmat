//! MATLAB-compatible triangulation objects and helpers.

use std::collections::HashMap;
use std::sync::OnceLock;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, PropertyDef,
};
use runmat_geometry_ops::{boundary_edges, delaunay_2d, nearest_neighbor_indices, point_locations};
use runmat_macros::runtime_builtin;
use runmat_value::{Access, IntegerStorage, NumericScalar, ObjectInstance, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
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

const INTEGER_COORDINATES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "delaunaytri-integer-coordinates",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "DelaunayTri construction, object points, or queries with typed-integer coordinates are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension"),
};

const INTEGER_TOPOLOGY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "delaunaytri-integer-topology",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "DelaunayTri constraints or object connectivity with typed-integer topology are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension"),
};

pub const CONSTRUCTOR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_COORDINATES_EXTENSION, INTEGER_TOPOLOGY_EXTENSION];
pub const COORDINATE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INTEGER_COORDINATES_EXTENSION];
pub const TOPOLOGY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INTEGER_TOPOLOGY_EXTENSION];
pub const POINT_LOCATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_COORDINATES_EXTENSION, INTEGER_TOPOLOGY_EXTENSION];

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

const CONSTRUCTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "coordinates",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer point matrices or coordinate vectors are gated by delaunaytri-integer-coordinates before entering the documented double geometry domain.",
    },
    BuiltinIntegerInputCapability {
        name: "constraints",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer constraint/connectivity identifiers are gated by delaunaytri-integer-topology and parsed exactly through platform bounds.",
    },
];

pub const CONSTRUCTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dt = DelaunayTri(X, C)",
        inputs: &CONSTRUCTOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed coordinates/topology are RunMat-only; geometry and public object properties use documented double arrays.",
    }];

const TOPOLOGY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dt.ConnectivityList",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer object connectivity is gated by delaunaytri-integer-topology and parsed exactly.",
    }];

pub const FREE_BOUNDARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "E = freeBoundary(dt)",
        inputs: &TOPOLOGY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed topology is a RunMat-only object state; returned boundary vertex indices are documented double.",
    }];

const COORDINATE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "coordinates",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer object/query coordinates are gated by delaunaytri-integer-coordinates before conversion to the double geometry domain.",
    }];

pub const NEAREST_NEIGHBOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "vi = nearestNeighbor(dt, q)",
        inputs: &COORDINATE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed coordinates are RunMat-only; nearest-vertex indices are documented double.",
    }];

const POINT_LOCATION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "coordinates",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer object/query coordinates are gated by delaunaytri-integer-coordinates.",
    },
    BuiltinIntegerInputCapability {
        name: "dt.ConnectivityList",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer object connectivity is gated by delaunaytri-integer-topology and parsed exactly.",
    },
];

pub const POINT_LOCATION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[ti, bc] = pointLocation(dt, q)",
        inputs: &POINT_LOCATION_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed coordinates/topology are RunMat-only; triangle indices and barycentric coordinates are documented double.",
    }];

#[runtime_builtin(
    name = "DelaunayTri",
    category = "geometry/triangulation",
    summary = "Create a legacy 2-D Delaunay triangulation object.",
    keywords = "DelaunayTri,delaunay,triangulation,geometry,mesh",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::DELAUNAY_TRI_DESCRIPTOR),
    extensions(crate::builtins::geometry::triangulation::CONSTRUCTOR_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::CONSTRUCTOR_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::geometry::triangulation"
)]
pub async fn delaunay_tri_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_registered();
    ensure_resident_constructor_extensions(&args)?;
    let parsed = parse_constructor_args(args).await?;
    ensure_coordinate_extension(parsed.integer_coordinates)?;
    ensure_topology_extension(parsed.integer_topology)?;
    Ok(Value::Object(delaunay_object(parsed)?))
}

#[runtime_builtin(
    name = "freeBoundary",
    category = "geometry/triangulation",
    summary = "Return the boundary edges of a DelaunayTri object.",
    keywords = "freeBoundary,DelaunayTri,triangulation,boundary,mesh",
    accel = "cpu",
    descriptor(crate::builtins::geometry::triangulation::HELPER_DESCRIPTOR),
    extensions(crate::builtins::geometry::triangulation::TOPOLOGY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::FREE_BOUNDARY_INTEGER_CAPABILITIES
    ),
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
    extensions(crate::builtins::geometry::triangulation::TOPOLOGY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::FREE_BOUNDARY_INTEGER_CAPABILITIES
    ),
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
    extensions(crate::builtins::geometry::triangulation::COORDINATE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::NEAREST_NEIGHBOR_INTEGER_CAPABILITIES
    ),
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
    extensions(crate::builtins::geometry::triangulation::COORDINATE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::geometry::triangulation::NEAREST_NEIGHBOR_INTEGER_CAPABILITIES
    ),
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
    integer_coordinates: bool,
    integer_topology: bool,
}

async fn parse_constructor_args(args: Vec<Value>) -> BuiltinResult<ParsedDelaunay> {
    let gathered = gather_values(args).await?;
    match gathered.as_slice() {
        [] => Ok(ParsedDelaunay {
            points: Vec::new(),
            constraints: Vec::new(),
            integer_coordinates: false,
            integer_topology: false,
        }),
        [points] => Ok(ParsedDelaunay {
            points: matrix_points(points, BUILTIN_NAME)?,
            constraints: Vec::new(),
            integer_coordinates: is_typed_integer_numeric_value(points),
            integer_topology: false,
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
                integer_coordinates: is_typed_integer_numeric_value(first),
                integer_topology: is_typed_integer_numeric_value(second),
            })
        }
        [x, y] => Ok(ParsedDelaunay {
            points: vector_points(x, y, BUILTIN_NAME)?,
            constraints: Vec::new(),
            integer_coordinates: is_typed_integer_numeric_value(x)
                || is_typed_integer_numeric_value(y),
            integer_topology: false,
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

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

fn ensure_resident_constructor_extensions(args: &[Value]) -> BuiltinResult<()> {
    match args {
        [points] => ensure_resident_coordinate_values(std::slice::from_ref(points)),
        [first, second] if looks_like_point_matrix(first) && !is_vector_value(first) => {
            ensure_resident_coordinate_values(std::slice::from_ref(first))?;
            ensure_resident_topology_values(std::slice::from_ref(second))
        }
        [x, y] => ensure_resident_coordinate_values(&[x.clone(), y.clone()]),
        [x, y, topology] => {
            ensure_resident_coordinate_values(&[x.clone(), y.clone()])?;
            ensure_resident_topology_values(std::slice::from_ref(topology))
        }
        _ => Ok(()),
    }
}

fn ensure_resident_coordinate_values(values: &[Value]) -> BuiltinResult<()> {
    if values.iter().any(is_resident_typed_integer_value) {
        ensure_coordinate_extension(true)?;
    }
    Ok(())
}

fn ensure_resident_topology_values(values: &[Value]) -> BuiltinResult<()> {
    if values.iter().any(is_resident_typed_integer_value) {
        ensure_topology_extension(true)?;
    }
    Ok(())
}

fn ensure_coordinate_extension(enabled: bool) -> BuiltinResult<()> {
    if enabled {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_COORDINATES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn ensure_topology_extension(enabled: bool) -> BuiltinResult<()> {
    if enabled {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_TOPOLOGY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_typed_integer_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || is_resident_typed_integer_value(value)
}

fn is_resident_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn object_coordinates_are_typed_integer(object: &ObjectInstance) -> bool {
    object
        .properties
        .get(POINTS_PROPERTY)
        .or_else(|| object.properties.get(LEGACY_POINTS_PROPERTY))
        .is_some_and(is_typed_integer_numeric_value)
}

fn object_topology_is_typed_integer(object: &ObjectInstance) -> bool {
    object
        .properties
        .get(CONNECTIVITY_PROPERTY)
        .or_else(|| object.properties.get(LEGACY_CONNECTIVITY_PROPERTY))
        .is_some_and(is_typed_integer_numeric_value)
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
    let integer_topology = object_topology_is_typed_integer(object);
    let triangles = object_triangles(object, None)?;
    ensure_topology_extension(integer_topology)?;
    let edges = boundary_edges(&triangles);
    edges_tensor(&edges)
}

async fn nearest_neighbor_value(dt: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let object = require_delaunay_object(&dt)?;
    let object_integer_coordinates = object_coordinates_are_typed_integer(object);
    let points = object_points(object)?;
    ensure_resident_coordinate_values(&rest)?;
    let gathered = gather_values(rest).await?;
    let query_integer_coordinates = gathered.iter().any(is_typed_integer_numeric_value);
    let queries = parse_query_points(gathered, "nearestNeighbor")?;
    ensure_coordinate_extension(object_integer_coordinates || query_integer_coordinates)?;
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
    let object_integer_coordinates = object_coordinates_are_typed_integer(object);
    let object_integer_topology = object_topology_is_typed_integer(object);
    let points = object_points(object)?;
    let triangles = object_triangles(object, Some(points.len()))?;
    ensure_resident_coordinate_values(&rest)?;
    let gathered = gather_values(rest).await?;
    let query_integer_coordinates = gathered.iter().any(is_typed_integer_numeric_value);
    let queries = parse_query_points(gathered, "pointLocation")?;
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
    if tensor.cols != 2 {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut points = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let x = values[row];
        let y = values[row + tensor.rows];
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
    if !is_vector(&x) || !is_vector(&y) || x.len() != y.len() {
        return Err(invalid(format!(
            "{builtin}: x and y must be vectors with the same length"
        )));
    }
    let x_values = tensor_utils::tensor_values_f64_cow(&x);
    let y_values = tensor_utils::tensor_values_f64_cow(&y);
    x_values
        .iter()
        .zip(y_values.iter())
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
    if tensor.is_empty() || tensor.rows == 0 {
        return Ok(Vec::new());
    }
    if tensor.cols != 2 {
        return Err(invalid("DelaunayTri: constraints must be an M-by-2 matrix"));
    }
    let mut constraints = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        constraints.push([
            tensor_positive_index(&tensor, row, usize::MAX)?,
            tensor_positive_index(&tensor, row + tensor.rows, usize::MAX)?,
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
            tensor_positive_index(&tensor, row, usize::MAX)? - 1,
            tensor_positive_index(&tensor, row + tensor.rows, usize::MAX)? - 1,
            tensor_positive_index(&tensor, row + 2 * tensor.rows, usize::MAX)? - 1,
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
        Value::Int(value) => {
            Tensor::new_integer(IntegerStorage::from_scalar(value.clone()), vec![1, 1])
                .map_err(invalid)
        }
        other => Err(invalid(format!(
            "{builtin}: expected numeric input, got {other:?}"
        ))),
    }
}

fn looks_like_point_matrix(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols >= 2)
        || matches!(value, Value::GpuTensor(handle) if handle.shape.get(1).copied().unwrap_or(1) >= 2)
}

fn looks_like_constraints(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.cols == 2)
}

fn is_vector_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if is_vector(tensor))
        || matches!(value, Value::GpuTensor(handle) if handle.shape.first().copied().unwrap_or(1) == 1 || handle.shape.get(1).copied().unwrap_or(1) == 1)
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
    Tensor::new_2d(data, triangles.len(), 3)
        .map(Value::Tensor)
        .map_err(invalid)
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

fn tensor_positive_index(tensor: &Tensor, index: usize, max: usize) -> BuiltinResult<usize> {
    let value = tensor
        .numeric_value_at(index)
        .ok_or_else(|| invalid("index is out of range"))?;
    if let Some(integer) = value.into_int_value() {
        let idx = integer
            .try_to_usize()
            .ok_or_else(|| invalid("index is out of range"))?;
        if idx == 0 {
            return Err(invalid("indices must be positive finite integers"));
        }
        if idx > max {
            return Err(invalid("index is out of range"));
        }
        return Ok(idx);
    }
    match value {
        NumericScalar::F64(value) => positive_index(value, max),
        NumericScalar::F32(value) => positive_index(f64::from(value), max),
        _ => unreachable!("integer scalar returned by into_int_value"),
    }
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
        Value::Tensor(Tensor::new_2d(data.to_vec(), rows, cols).expect("test tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor"))
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
    fn delaunaytri_rejects_point_matrices_that_are_not_two_dimensional() {
        let err = block_on(delaunay_tri_builtin(vec![tensor(
            &[0.0, 1.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0],
            2,
            4,
        )]))
        .expect_err("4-column point matrix should fail");
        assert!(err.message.contains("N-by-2"));
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
    fn point_parsers_read_typed_integer_storage_exactly() {
        let points = matrix_points(
            &integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2),
            BUILTIN_NAME,
        )
        .expect("matrix points");
        assert_eq!(points, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let vector_points = vector_points(
            &integer_tensor(IntegerStorage::I16(vec![0, 1, 0]), 1, 3),
            &integer_tensor(IntegerStorage::I16(vec![0, 0, 1]), 1, 3),
            BUILTIN_NAME,
        )
        .expect("vector points");
        assert_eq!(vector_points, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn topology_parsers_read_typed_integer_storage_exactly() {
        let constraints =
            constraints_from_value(&integer_tensor(IntegerStorage::U16(vec![1, 2, 2, 3]), 2, 2))
                .expect("constraints");
        assert_eq!(constraints, vec![[1, 2], [2, 3]]);

        let connectivity = connectivity_from_value(
            &integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
            Some(3),
        )
        .expect("connectivity");
        assert_eq!(connectivity, vec![[0, 1, 2]]);
    }

    #[test]
    fn delaunaytri_integer_extensions_follow_compatibility_mode() {
        let points = || integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2);
        let empty_topology = || integer_tensor(IntegerStorage::U16(Vec::new()), 0, 2);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(delaunay_tri_builtin(vec![points()]))
                .expect_err("MATLAB mode rejects typed-integer constructor coordinates");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
            );

            let error = block_on(delaunay_tri_builtin(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                empty_topology(),
            ]))
            .expect_err("MATLAB mode rejects typed-integer constructor topology");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
            );

            let resident_points = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![3, 2],
                device_id: 0,
                buffer_id: 9_307_001,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &resident_points,
                runmat_accelerate_api::IntegerElementType::U16,
            );
            let error = block_on(delaunay_tri_builtin(vec![Value::GpuTensor(
                resident_points.clone(),
            )]))
            .expect_err("MATLAB mode rejects resident integer coordinates before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
            );
            runmat_accelerate_api::clear_handle_integer_type(&resident_points);

            let resident_topology = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![0, 2],
                device_id: 0,
                buffer_id: 9_307_002,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &resident_topology,
                runmat_accelerate_api::IntegerElementType::U16,
            );
            let error = block_on(delaunay_tri_builtin(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                Value::GpuTensor(resident_topology.clone()),
            ]))
            .expect_err("MATLAB mode rejects resident integer topology before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
            );
            runmat_accelerate_api::clear_handle_integer_type(&resident_topology);
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(block_on(delaunay_tri_builtin(vec![points()])).is_ok());
            assert!(block_on(delaunay_tri_builtin(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                empty_topology(),
            ]))
            .is_ok());
        }
    }

    #[test]
    fn delaunaytri_object_and_query_extensions_follow_compatibility_mode() {
        let make_object = || {
            object(
                block_on(delaunay_tri_builtin(vec![tensor(
                    &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    3,
                    2,
                )]))
                .expect("double constructor"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

            let mut integer_points = make_object();
            integer_points.properties.insert(
                POINTS_PROPERTY.to_string(),
                integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2),
            );
            let error = block_on(nearest_neighbor_builtin(
                Value::Object(integer_points),
                vec![tensor(&[0.1, 0.1], 1, 2)],
            ))
            .expect_err("MATLAB mode rejects integer object points");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
            );

            let object = Value::Object(make_object());
            let error = block_on(nearest_neighbor_builtin(
                object,
                vec![integer_tensor(IntegerStorage::U16(vec![0, 0]), 1, 2)],
            ))
            .expect_err("MATLAB mode rejects integer query points");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
            );

            let mut integer_topology = make_object();
            integer_topology.properties.insert(
                CONNECTIVITY_PROPERTY.to_string(),
                integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
            );
            let error = block_on(free_boundary_builtin(Value::Object(integer_topology)))
                .expect_err("MATLAB mode rejects integer object topology");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let object = Value::Object(make_object());
            assert!(block_on(nearest_neighbor_builtin(
                object,
                vec![integer_tensor(IntegerStorage::U16(vec![0, 0]), 1, 2)],
            ))
            .is_ok());

            let mut integer_topology = make_object();
            integer_topology.properties.insert(
                CONNECTIVITY_PROPERTY.to_string(),
                integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
            );
            assert!(block_on(free_boundary_builtin(Value::Object(integer_topology))).is_ok());
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn topology_index_parser_preserves_wide_integer_values_exactly() {
        let wide = (1_u64 << 53) + 1;
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide index");
        assert_eq!(
            tensor_positive_index(&tensor, 0, usize::MAX).expect("exact wide index"),
            wide as usize
        );
        assert!(tensor_positive_index(&tensor, 0, (wide - 1) as usize).is_err());
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
        assert_eq!(nn.materialize_f64(), vec![3.0]);

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
        assert_eq!(ti.materialize_f64(), vec![1.0]);
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
