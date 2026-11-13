use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or logical RGB image with replicated grayscale channels.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale input image.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = gray2rgb(I)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const GRAY2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:gray2rgb:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "gray2rgb: too many input arguments",
};
pub const GRAY2RGB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gray2rgb:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "gray2rgb: too many output arguments",
};
pub const GRAY2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.INVALID_INPUT",
    identifier: Some("RunMat:gray2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxN grayscale image.",
    message: "gray2rgb: invalid input",
};
pub const GRAY2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.INTERNAL",
    identifier: Some("RunMat:gray2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "gray2rgb: internal conversion failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GRAY2RGB_ERROR_TOO_MANY_INPUTS,
    GRAY2RGB_ERROR_TOO_MANY_OUTPUTS,
    GRAY2RGB_ERROR_INVALID_INPUT,
    GRAY2RGB_ERROR_INTERNAL,
];
pub const GRAY2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const GRAY2RGB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BuiltinExtensionDescriptor {
    id: "gray2rgb-callable",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "gray2rgb is a RunMat-provided image helper rather than an API in the compatibility target",
    error_identifier: Some("RunMat:compatibility:Gray2rgbCallExtension"),
}];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "I",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are replicated with exact source class and values preserved.",
}];
pub const GRAY2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "RGB = gray2rgb(integer_I)", inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The callable is RunMat-only. Exact integer planes are replicated and resident inputs gather to a host result.",
    },
];
