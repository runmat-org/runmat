use crate::*;

pub const DET_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "det-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "det with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DetIntegerInputExtension"),
};
pub const DET_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "det-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "det with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DetLogicalInputExtension"),
};
pub const DET_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [DET_INTEGER_INPUT_EXTENSION, DET_LOGICAL_INPUT_EXTENSION];
pub const DET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents square single/double matrices; RunMat mode additionally admits all eight real integer classes.",
}];
pub const DET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "d = det(integer_A)",
        inputs: &DET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Authoritative integer values enter one explicit binary64 LU boundary. Resident integer input gathers exactly before LU and restores the double scalar to the owning provider.",
    }];

pub const DET_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Determinant of A.",
}];

pub const DET_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const DET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "d = det(A)",
    inputs: &DET_INPUTS,
    outputs: &DET_OUTPUT,
}];

pub const DET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.INVALID_INPUT",
    identifier: Some("RunMat:det:InvalidInput"),
    when: "Input shape/type or numeric domain is unsupported for determinant evaluation.",
    message: "det: invalid input",
};

pub const DET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.INTERNAL",
    identifier: Some("RunMat:det:Internal"),
    when: "Runtime fails while evaluating determinant or device fallback paths.",
    message: "det: internal runtime failure",
};
pub const DET_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:det:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "det: too many output arguments",
};

pub const DET_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DET_ERROR_INVALID_INPUT,
    DET_ERROR_INTERNAL,
    DET_ERROR_TOO_MANY_OUTPUTS,
];

pub const DET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DET_ERRORS,
};
