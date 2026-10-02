use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale image converted from RGB input.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image or double colormap.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "I = rgb2gray(RGB)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
pub const RGB2GRAY_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2gray:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2gray: too many input arguments",
};
pub const RGB2GRAY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rgb2gray:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "rgb2gray: too many output arguments",
};
pub const RGB2GRAY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INVALID_INPUT",
    identifier: Some("RunMat:rgb2gray:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or an Nx3 double colormap.",
    message: "rgb2gray: invalid input",
};
pub const RGB2GRAY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INTERNAL",
    identifier: Some("RunMat:rgb2gray:Internal"),
    when: "Grayscale output tensor construction fails internally.",
    message: "rgb2gray: internal conversion failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    RGB2GRAY_ERROR_TOO_MANY_INPUTS,
    RGB2GRAY_ERROR_TOO_MANY_OUTPUTS,
    RGB2GRAY_ERROR_INVALID_INPUT,
    RGB2GRAY_ERROR_INTERNAL,
];
pub const RGB2GRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const DOCUMENTED: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "RGB",
    classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "uint8 and uint16 truecolor images preserve their native integer class.",
}];
const REJECTED: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "RGB",
    classes: &[
        BuiltinIntegerClass::Int8,
        BuiltinIntegerClass::Int16,
        BuiltinIntegerClass::Int32,
        BuiltinIntegerClass::Int64,
        BuiltinIntegerClass::Uint32,
        BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Signed and wide unsigned RGB classes reject before resident download.",
}];
pub const RGB2GRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "I = rgb2gray(integer_RGB)", inputs: &DOCUMENTED,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8/uint16 samples use luminance coefficients and nearest ties-away rounding with saturation; resident results restore through their exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2gray(unsupported_integer_RGB)", inputs: &REJECTED,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];
