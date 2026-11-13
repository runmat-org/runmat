use crate::*;

const RGB2LAB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LAB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "CIE L*a*b* image or colormap converted from RGB input.",
}];

const RGB2LAB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "RGB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RGB image, image stack, or c-by-3 colormap values.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ColorSpace or WhitePoint name-value options.",
    },
];

const RGB2LAB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "LAB = rgb2lab(RGB, Name=Value)",
    inputs: &RGB2LAB_INPUTS,
    outputs: &RGB2LAB_OUTPUT,
}];

pub const RGB2LAB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2lab:TooManyInputs"),
    when: "More than the two supported name-value pairs are supplied.",
    message: "rgb2lab: too many input arguments",
};

pub const RGB2LAB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rgb2lab:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "rgb2lab: too many output arguments",
};

pub const RGB2LAB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INVALID_INPUT",
    identifier: Some("RunMat:rgb2lab:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or Nx3 RGB colormap.",
    message: "rgb2lab: invalid input",
};

pub const RGB2LAB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INTERNAL",
    identifier: Some("RunMat:rgb2lab:Internal"),
    when: "L*a*b* output tensor construction fails internally.",
    message: "rgb2lab: internal conversion failure",
};

pub const RGB2LAB_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INVALID_OPTION",
    identifier: Some("RunMat:rgb2lab:InvalidOption"),
    when: "A ColorSpace or WhitePoint name-value argument is malformed or unsupported.",
    message: "rgb2lab: invalid name-value option",
};

const RGB2LAB_ERRORS: [BuiltinErrorDescriptor; 5] = [
    RGB2LAB_ERROR_TOO_MANY_OUTPUTS,
    RGB2LAB_ERROR_TOO_MANY_INPUTS,
    RGB2LAB_ERROR_INVALID_INPUT,
    RGB2LAB_ERROR_INTERNAL,
    RGB2LAB_ERROR_INVALID_OPTION,
];

pub const RGB2LAB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2LAB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2LAB_ERRORS,
};

const RGB2LAB_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rgb2lab-explicit-gpu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rgb2lab with an explicitly requested gpuArray input is a RunMat extension because the compatibility target documents no interactive GPU Arrays capability",
    error_identifier: Some("RunMat:compatibility:Rgb2labExplicitGpuExtension"),
};
pub const RGB2LAB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RGB2LAB_EXPLICIT_GPU_EXTENSION];

const RGB2LAB_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "uint8 and uint16 RGB images, MxNx3xP stacks, and c-by-3 colormaps are documented and scale to double L*a*b* output.",
    }];
const RGB2LAB_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &[BuiltinIntegerClass::Int8, BuiltinIntegerClass::Int16, BuiltinIntegerClass::Int32, BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint32, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Signed integer and uint32/uint64 RGB values are outside the documented surface and reject before resident download.",
    }];
pub const RGB2LAB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "LAB = rgb2lab(integer_RGB)",
        inputs: &RGB2LAB_DOCUMENTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8/uint16 samples are normalized by their full class range before the documented floating color-space conversion and produce double output. Automatic residency may gather to the documented host implementation; the separately gated explicit-gpuArray extension restores output to the exact owner or errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2lab(unsupported_integer_RGB)",
        inputs: &RGB2LAB_REJECTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];
