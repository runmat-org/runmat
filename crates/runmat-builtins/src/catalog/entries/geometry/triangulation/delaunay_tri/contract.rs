use crate::*;

pub const CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("DelaunayTri");

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-dimensional DelaunayTri object with one-based connectivity.",
}];
const POINTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite N-by-2 point matrix.",
}];
const COORDINATES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite x-coordinate vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite y-coordinate vector of the same length.",
    },
];
const POINTS_CONSTRAINTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite N-by-2 point matrix.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Empty M-by-2 constraints; nonempty constraints are unsupported.",
    },
];
const VECTORS_CONSTRAINTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite x-coordinate vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite y-coordinate vector of the same length.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Empty M-by-2 constraints; nonempty constraints are unsupported.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri()",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(X)",
        inputs: &POINTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(x, y)",
        inputs: &COORDINATES,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(X, C)",
        inputs: &POINTS_CONSTRAINTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dt = DelaunayTri(x, y, C)",
        inputs: &VECTORS_CONSTRAINTS,
        outputs: &OUTPUT,
    },
];
pub const TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELAUNAY_TRI.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:DelaunayTri:TooManyOutputs"),
    when: "More than one constructor result is requested.",
    message: "DelaunayTri: too many output arguments",
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

pub const COORDINATES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "delaunaytri-integer-coordinates",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "DelaunayTri construction, object points, or queries with typed-integer coordinates are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension"),
};
pub const COORDINATE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COORDINATES_EXTENSION];
pub const CONSTRUCTOR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    COORDINATES_EXTENSION,
    super::super::free_boundary::DELAUNAY_TRI_INTEGER_TOPOLOGY_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "coordinates",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer points and coordinate vectors enter the double geometry domain only in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "constraints",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer connectivity is parsed exactly through platform bounds and gated before gathering.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "dt = DelaunayTri(X, C)",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Typed coordinate and topology forms are RunMat-only; public object numeric properties are double.",
}];
