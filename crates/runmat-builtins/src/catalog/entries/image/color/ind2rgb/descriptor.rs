use crate::*;

const IND2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image converted from indexed image and colormap.",
}];

const IND2RGB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image values.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nx3 colormap.",
    },
];

const IND2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = ind2rgb(X, map)",
    inputs: &IND2RGB_INPUTS,
    outputs: &IND2RGB_OUTPUT,
}];

pub const IND2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:ind2rgb:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "ind2rgb: too many input arguments",
};

pub const IND2RGB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ind2rgb:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "ind2rgb: too many output arguments",
};

pub const IND2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_INPUT",
    identifier: Some("RunMat:ind2rgb:InvalidInput"),
    when: "Inputs cannot be interpreted as numeric indexed image and numeric colormap tensors.",
    message: "ind2rgb: invalid input",
};

pub const IND2RGB_ERROR_INVALID_COLORMAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_COLORMAP",
    identifier: Some("RunMat:ind2rgb:InvalidColormap"),
    when: "The map argument is not an Nx3 colormap.",
    message: "ind2rgb: map must be an Nx3 colormap",
};

pub const IND2RGB_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INVALID_INDEX",
    identifier: Some("RunMat:ind2rgb:InvalidIndex"),
    when: "At least one floating index value is not finite or integral.",
    message: "ind2rgb: floating index values must be finite integers",
};

pub const IND2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2RGB.INTERNAL",
    identifier: Some("RunMat:ind2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "ind2rgb: internal conversion failure",
};

const IND2RGB_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IND2RGB_ERROR_TOO_MANY_OUTPUTS,
    IND2RGB_ERROR_TOO_MANY_INPUTS,
    IND2RGB_ERROR_INVALID_INPUT,
    IND2RGB_ERROR_INVALID_COLORMAP,
    IND2RGB_ERROR_INVALID_INDEX,
    IND2RGB_ERROR_INTERNAL,
];

pub const IND2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IND2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2RGB_ERRORS,
};

const IND2RGB_DOCUMENTED_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "uint8 and uint16 indexed images use documented zero-based indices and are clipped to the colormap extent before lookup.",
    }];
const IND2RGB_REJECTED_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[BuiltinIntegerClass::Int8, BuiltinIntegerClass::Int16, BuiltinIntegerClass::Int32, BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint32, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Other native integer classes are outside the documented indexed-image surface and reject before a resident value is downloaded.",
    }];
const IND2RGB_REJECTED_MAP_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "map",
        classes: &[BuiltinIntegerClass::Int8, BuiltinIntegerClass::Int16, BuiltinIntegerClass::Int32, BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16, BuiltinIntegerClass::Uint32, BuiltinIntegerClass::Uint64],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented colormap is double; typed-integer colormaps reject before lookup or resident download.",
    }];
pub const IND2RGB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "RGB = ind2rgb(integer_X, map)",
        inputs: &IND2RGB_DOCUMENTED_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Authoritative uint8/uint16 indices are zero-based and clipped exactly; documented GPU input is downloaded non-destructively and the double RGB result is restored through its owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ind2rgb(unsupported_integer_X, map)",
        inputs: &IND2RGB_REJECTED_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Unsupported indexed-image classes reject consistently on host and from resident dtype metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ind2rgb(X, integer_map)",
        inputs: &IND2RGB_REJECTED_MAP_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All typed-integer colormaps reject because the documented map class is double.",
    },
];
