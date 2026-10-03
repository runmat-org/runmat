use crate::*;

const RECEIVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "DelaunayTri object with host connectivity.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "E",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based boundary edges as an M-by-2 double array.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "E = freeBoundary(dt)",
    inputs: &RECEIVER,
    outputs: &OUTPUT,
}];
const METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "E = dt.freeBoundary()",
    inputs: &RECEIVER,
    outputs: &OUTPUT,
}];

pub const INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELAUNAY_TRI.INVALID_ARGUMENT",
    identifier: Some("RunMat:DelaunayTri:InvalidArgument"),
    when: "Constructor, object, constraint, or query inputs are malformed.",
    message: "DelaunayTri: invalid argument",
};
pub const UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELAUNAY_TRI.UNSUPPORTED",
    identifier: Some("RunMat:DelaunayTri:Unsupported"),
    when: "A requested triangulation form is not implemented.",
    message: "DelaunayTri: unsupported form",
};
pub const TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREE_BOUNDARY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:freeBoundary:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "freeBoundary: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [INVALID_ARGUMENT, UNSUPPORTED, TOO_MANY_OUTPUTS];
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

pub const TOPOLOGY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "delaunaytri-integer-topology",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "DelaunayTri constraints or object connectivity with typed-integer topology are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [TOPOLOGY_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dt.ConnectivityList",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed-integer object connectivity is gated by delaunaytri-integer-topology and parsed exactly.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "E = freeBoundary(dt)",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Typed topology is a RunMat-only object state; returned boundary vertex indices are double.",
}];
