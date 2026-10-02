use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "HSV",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HSV image or colormap converted from RGB input.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image or double Nx3 colormap.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "HSV = rgb2hsv(RGB)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
pub const RGB2HSV_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2hsv:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2hsv: too many input arguments",
};
pub const RGB2HSV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rgb2hsv:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "rgb2hsv: too many output arguments",
};
pub const RGB2HSV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INVALID_INPUT",
    identifier: Some("RunMat:rgb2hsv:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or Nx3 RGB colormap.",
    message: "rgb2hsv: invalid input",
};
pub const RGB2HSV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INTERNAL",
    identifier: Some("RunMat:rgb2hsv:Internal"),
    when: "HSV output tensor construction fails internally.",
    message: "rgb2hsv: internal conversion failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    RGB2HSV_ERROR_TOO_MANY_INPUTS,
    RGB2HSV_ERROR_TOO_MANY_OUTPUTS,
    RGB2HSV_ERROR_INVALID_INPUT,
    RGB2HSV_ERROR_INTERNAL,
];
pub const RGB2HSV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
    notes: "uint8 and uint16 truecolor images scale to double HSV output.",
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
pub const RGB2HSV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "HSV = rgb2hsv(integer_RGB)", inputs: &DOCUMENTED,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8/uint16 samples normalize before color conversion; documented resident input restores through its exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2hsv(unsupported_integer_RGB)", inputs: &REJECTED,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];
