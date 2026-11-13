use crate::*;

pub const INV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Inverse of A.",
}];

pub const INV_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input square matrix.",
}];

pub const INV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = inv(A)",
    inputs: &INV_INPUTS,
    outputs: &INV_OUTPUT,
}];

pub const INV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INV.INVALID_INPUT",
    identifier: Some("RunMat:inv:InvalidInput"),
    when: "Input shape/type is unsupported or matrix is singular for inversion.",
    message: "inv: invalid input",
};

pub const INV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INV.INTERNAL",
    identifier: Some("RunMat:inv:Internal"),
    when: "Runtime fails while executing inversion or fallback/upload paths.",
    message: "inv: internal runtime failure",
};

pub const INV_ERRORS: [BuiltinErrorDescriptor; 2] = [INV_ERROR_INVALID_INPUT, INV_ERROR_INTERNAL];

pub const INV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INV_ERRORS,
};

pub const INV_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inv-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inv with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InvIntegerInputExtension"),
};
pub const INV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inv with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InvLogicalInputExtension"),
};
pub const INV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INV_INTEGER_INPUT_EXTENSION, INV_LOGICAL_INPUT_EXTENSION];
pub const INV_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Every element must be exactly representable before the binary64 matrix-inversion boundary.",
}];
pub const INV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "X = inv(integer_A)",
        inputs: &INV_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only integer matrices cross one checked binary64 boundary. Resident integer input is classified before provider execution and the double result is restored only when the owning provider can represent it truthfully.",
    }];
