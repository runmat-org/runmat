use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB image or colormap converted from HSV input.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "HSV",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HSV image or Nx3 HSV colormap values.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = hsv2rgb(HSV)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
pub const HSV2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:hsv2rgb:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "hsv2rgb: too many input arguments",
};
pub const HSV2RGB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:hsv2rgb:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "hsv2rgb: too many output arguments",
};
pub const HSV2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.INVALID_INPUT",
    identifier: Some("RunMat:hsv2rgb:InvalidInput"),
    when: "Input is not an admitted HSV image or colormap class and geometry.",
    message: "hsv2rgb: invalid input",
};
pub const HSV2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.INTERNAL",
    identifier: Some("RunMat:hsv2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "hsv2rgb: internal conversion failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    HSV2RGB_ERROR_TOO_MANY_INPUTS,
    HSV2RGB_ERROR_TOO_MANY_OUTPUTS,
    HSV2RGB_ERROR_INVALID_INPUT,
    HSV2RGB_ERROR_INTERNAL,
];
pub const HSV2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const REJECTED: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "HSV", classes: &[
        BuiltinIntegerClass::Int8, BuiltinIntegerClass::Int16,
        BuiltinIntegerClass::Int32, BuiltinIntegerClass::Int64,
        BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16,
        BuiltinIntegerClass::Uint32, BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The HSV surface accepts single, double, or logical images and single or double colormaps; every typed integer class rejects.",
}];
pub const HSV2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "hsv2rgb(integer_HSV)",
        inputs: &REJECTED,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host integer storage and resident integer dtype metadata reject before download.",
    }];
