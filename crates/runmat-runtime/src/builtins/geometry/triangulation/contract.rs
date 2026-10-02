use super::*;

pub const DELAUNAY_TRI_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("DelaunayTri");
pub(super) const BUILTIN_NAME: &str = "DelaunayTri";
pub(super) const POINTS_PROPERTY: &str = "Points";
pub(super) const CONNECTIVITY_PROPERTY: &str = "ConnectivityList";
pub(super) const LEGACY_POINTS_PROPERTY: &str = "X";
pub(super) const LEGACY_CONNECTIVITY_PROPERTY: &str = "Triangulation";
pub(super) const CONSTRAINTS_PROPERTY: &str = "Constraints";
pub(super) const DIMENSION_PROPERTY: &str = "Dimension";

pub(super) const INTEGER_COORDINATES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "delaunaytri-integer-coordinates",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "DelaunayTri construction, object points, or queries with typed-integer coordinates are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension"),
};

pub(super) const INTEGER_TOPOLOGY_EXTENSION: BuiltinExtensionDescriptor =
    runmat_builtins::DELAUNAY_TRI_INTEGER_TOPOLOGY_EXTENSION;

pub const CONSTRUCTOR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_COORDINATES_EXTENSION, INTEGER_TOPOLOGY_EXTENSION];
pub const COORDINATE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INTEGER_COORDINATES_EXTENSION];
pub const POINT_LOCATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_COORDINATES_EXTENSION, INTEGER_TOPOLOGY_EXTENSION];

pub(super) const OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy DelaunayTri triangulation object.",
}];

pub(super) const INPUT_VARARGIN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Point matrix, coordinate vectors, and optional constraints.",
}];

pub(super) const DELAUNAY_TRI_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
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

pub(super) const OUTPUT_TENSOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric result.",
}];

pub(super) const OUTPUT_POINT_LOCATION: [BuiltinParamDescriptor; 2] = [
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

pub(super) const INPUT_OBJECT_REST: [BuiltinParamDescriptor; 2] = [
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

pub(super) const HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = helper(dt, varargin)",
        inputs: &INPUT_OBJECT_REST,
        outputs: &OUTPUT_TENSOR,
    }];

pub(super) const POINT_LOCATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "[ti, bc] = pointLocation(dt, q)",
        inputs: &INPUT_OBJECT_REST,
        outputs: &OUTPUT_POINT_LOCATION,
    }];

pub(super) const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    runmat_builtins::DELAUNAY_TRI_ERROR_INVALID_ARGUMENT;

pub(super) const ERROR_UNSUPPORTED: BuiltinErrorDescriptor =
    runmat_builtins::DELAUNAY_TRI_ERROR_UNSUPPORTED;

pub(super) const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_UNSUPPORTED];

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

pub(super) const CONSTRUCTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
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

pub(super) const COORDINATE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
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

pub(super) const POINT_LOCATION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
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
