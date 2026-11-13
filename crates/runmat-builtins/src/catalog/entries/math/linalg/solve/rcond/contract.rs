use crate::*;

pub const RCOND_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rcond-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rcond with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RcondIntegerInputExtension"),
};

pub const RCOND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rcond-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rcond with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RcondLogicalInputExtension"),
};

pub const RCOND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RCOND_INTEGER_INPUT_EXTENSION, RCOND_LOGICAL_INPUT_EXTENSION];

pub const RCOND_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public matrix classes are single and double; RunMat mode admits real integer matrices only when every value is exactly representable as binary64.",
    }];

pub const RCOND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "c = rcond(integer_A)",
        inputs: &RCOND_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated integer matrix crosses one checked binary64 condition-estimation boundary. Host output is double, and automatic residency may restore the floating scalar to the owner.",
    }];

pub const RCOND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reciprocal condition number estimate.",
}];

pub const RCOND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input square matrix.",
}];

pub const RCOND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "c = rcond(A)",
    inputs: &RCOND_INPUTS,
    outputs: &RCOND_OUTPUT,
}];

pub const RCOND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RCOND.INVALID_INPUT",
    identifier: Some("RunMat:rcond:InvalidInput"),
    when: "Input shape/type is unsupported for reciprocal condition estimation.",
    message: "rcond: invalid input",
};

pub const RCOND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RCOND.INTERNAL",
    identifier: Some("RunMat:rcond:Internal"),
    when: "Runtime fails while computing rcond or executing fallback/upload paths.",
    message: "rcond: internal runtime failure",
};

pub const RCOND_ERRORS: [BuiltinErrorDescriptor; 2] =
    [RCOND_ERROR_INVALID_INPUT, RCOND_ERROR_INTERNAL];

pub const RCOND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RCOND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RCOND_ERRORS,
};
