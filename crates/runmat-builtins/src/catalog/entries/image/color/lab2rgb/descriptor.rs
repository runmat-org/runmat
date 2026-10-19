use crate::*;

const LAB2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "sRGB image or colormap converted from CIE L*a*b* input.",
}];

const LAB2RGB_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "LAB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CIE L*a*b* image or Nx3 colormap values.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "OutputType name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "double, single, uint8, or uint16 output class.",
    },
];

const LAB2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = lab2rgb(LAB, OutputType=type)",
    inputs: &LAB2RGB_INPUTS,
    outputs: &LAB2RGB_OUTPUT,
}];

pub const LAB2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:lab2rgb:TooManyInputs"),
    when: "Too many name-value arguments are supplied.",
    message: "lab2rgb: too many input arguments",
};

pub const LAB2RGB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:lab2rgb:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "lab2rgb: too many output arguments",
};

pub const LAB2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INVALID_INPUT",
    identifier: Some("RunMat:lab2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 L*a*b* image or Nx3 colormap.",
    message: "lab2rgb: invalid input",
};

pub const LAB2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INTERNAL",
    identifier: Some("RunMat:lab2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "lab2rgb: internal conversion failure",
};
pub const LAB2RGB_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INVALID_OPTION",
    identifier: Some("RunMat:lab2rgb:InvalidOption"),
    when: "OutputType is malformed or unsupported.",
    message: "lab2rgb: invalid option",
};

const LAB2RGB_ERRORS: [BuiltinErrorDescriptor; 5] = [
    LAB2RGB_ERROR_TOO_MANY_OUTPUTS,
    LAB2RGB_ERROR_TOO_MANY_INPUTS,
    LAB2RGB_ERROR_INVALID_INPUT,
    LAB2RGB_ERROR_INTERNAL,
    LAB2RGB_ERROR_INVALID_OPTION,
];

pub const LAB2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LAB2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LAB2RGB_ERRORS,
};

const LAB2RGB_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lab2rgb-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lab2rgb with an explicitly requested gpuArray input is a RunMat extension because the compatibility target documents no interactive GPU Arrays capability",
    error_identifier: Some("RunMat:compatibility:Lab2rgbExplicitGpuExtension"),
};
pub const LAB2RGB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LAB2RGB_EXPLICIT_GPU_EXTENSION];

const LAB2RGB_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "LAB",
    classes: &[BuiltinIntegerClass::Int8, BuiltinIntegerClass::Int16, BuiltinIntegerClass::Int32, BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16, BuiltinIntegerClass::Uint32, BuiltinIntegerClass::Uint64],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target accepts only single or double LAB input. Every integer class rejects from host or resident metadata before conversion or gather.",
}];
pub const LAB2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "RGB = lab2rgb(LAB, OutputType=integer_type)",
    inputs: &LAB2RGB_INTEGER_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::OptionDependent,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Integer LAB input is rejected. For floating LAB input, OutputType uint8 or uint16 clamps normalized RGB to [0,1], rounds after scaling by the class maximum, and constructs authoritative native integer storage; floating output preserves out-of-gamut values. Automatic residency may gather transparently, while the gated explicit-gpuArray extension restores output to the exact owner or errors.",
}];
