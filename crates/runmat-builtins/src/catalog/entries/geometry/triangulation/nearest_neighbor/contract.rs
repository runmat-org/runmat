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
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "vi",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based nearest-vertex indices, or NaN if there are no points.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "vi = nearestNeighbor(dt,q)",
        inputs: &QUERY_MATRIX,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "vi = nearestNeighbor(dt,x,y)",
        inputs: &QUERY_VECTORS,
        outputs: &OUTPUT,
    },
];
const METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "vi = dt.nearestNeighbor(q)",
        inputs: &QUERY_MATRIX,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "vi = dt.nearestNeighbor(x,y)",
        inputs: &QUERY_VECTORS,
        outputs: &OUTPUT,
    },
];
pub const TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEAREST_NEIGHBOR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:nearestNeighbor:TooManyOutputs"),
    when: "More than one nearest-neighbor output is requested.",
    message: "nearestNeighbor: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    super::super::free_boundary::DELAUNAY_TRI_ERROR_INVALID_ARGUMENT,
    super::super::free_boundary::DELAUNAY_TRI_ERROR_UNSUPPORTED,
    TOO_MANY_OUTPUTS,
];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "coordinates",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Typed-integer object and query coordinates are gated before conversion to double geometry.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "vi = nearestNeighbor(dt,q)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed coordinate inputs are RunMat-only; nearest-vertex indices are double.",
    }];
