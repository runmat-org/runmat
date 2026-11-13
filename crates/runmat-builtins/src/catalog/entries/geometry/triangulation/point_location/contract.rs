use crate::*;

const QUERY_MATRIX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "DelaunayTri object.",
    },
    BuiltinParamDescriptor {
        name: "q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite N-by-2 query coordinates.",
    },
];
const QUERY_VECTORS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "DelaunayTri object.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query x-coordinate vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query y-coordinate vector of the same length.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ti",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based containing-triangle indices, or NaN for outside points.",
    },
    BuiltinParamDescriptor {
        name: "bc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "N-by-3 barycentric coordinates when two outputs are requested.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[ti,bc] = pointLocation(dt,q)",
        inputs: &QUERY_MATRIX,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ti,bc] = pointLocation(dt,x,y)",
        inputs: &QUERY_VECTORS,
        outputs: &OUTPUTS,
    },
];
const METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[ti,bc] = dt.pointLocation(q)",
        inputs: &QUERY_MATRIX,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ti,bc] = dt.pointLocation(x,y)",
        inputs: &QUERY_VECTORS,
        outputs: &OUTPUTS,
    },
];
pub const TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POINT_LOCATION.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pointLocation:TooManyOutputs"),
    when: "More than two point-location outputs are requested.",
    message: "pointLocation: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    super::super::free_boundary::DELAUNAY_TRI_ERROR_INVALID_ARGUMENT,
    super::super::free_boundary::DELAUNAY_TRI_ERROR_UNSUPPORTED,
    TOO_MANY_OUTPUTS,
];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    super::super::delaunay_tri::DELAUNAY_TRI_INTEGER_COORDINATES_EXTENSION,
    super::super::free_boundary::DELAUNAY_TRI_INTEGER_TOPOLOGY_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "coordinates", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer object and query coordinates are gated by delaunaytri-integer-coordinates." },
    BuiltinIntegerInputCapability { name: "dt.ConnectivityList", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Typed-integer object connectivity is gated by delaunaytri-integer-topology and parsed exactly." },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "[ti,bc] = pointLocation(dt,q)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Typed coordinate/topology inputs are RunMat-only; indices and barycentric coordinates are double.",
}];
