use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Window column vector.",
}];
const LENGTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative scalar window length.",
};
const SAMPLING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"symmetric\""),
    description: "Symmetric or periodic sampling.",
};
const PRECISION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "typeName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"double\""),
    description: "Double or single output precision.",
};
const INPUT_ONE: [BuiltinParamDescriptor; 1] = [LENGTH];
const INPUT_TWO_SAMPLING: [BuiltinParamDescriptor; 2] = [LENGTH, SAMPLING];
const INPUT_TWO_PRECISION: [BuiltinParamDescriptor; 2] = [LENGTH, PRECISION];
const INPUT_THREE: [BuiltinParamDescriptor; 3] = [LENGTH, SAMPLING, PRECISION];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "w = hann(n)",
        inputs: &INPUT_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, sampling)",
        inputs: &INPUT_TWO_SAMPLING,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, precision)",
        inputs: &INPUT_TWO_PRECISION,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, sampling, precision)",
        inputs: &INPUT_THREE,
        outputs: &OUTPUT,
    },
];
pub const HANN_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INVALID_LENGTH",
    identifier: Some("RunMat:hann:InvalidLength"),
    when: "Length is not a supported nonnegative scalar.",
    message: "hann: expected a nonnegative scalar integer length",
};
pub const HANN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INVALID_OPTION",
    identifier: Some("RunMat:hann:InvalidOption"),
    when: "An option is not a scalar text token.",
    message: "hann: unrecognized option",
};
pub const HANN_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.UNKNOWN_OPTION",
    identifier: Some("RunMat:hann:UnknownOption"),
    when: "An option is unrecognized or out of position.",
    message: "hann: unrecognized option",
};
pub const HANN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INTERNAL",
    identifier: Some("RunMat:hann:InternalError"),
    when: "Window materialization fails.",
    message: "hann: internal error",
};
pub const HANN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.ARG_COUNT",
    identifier: Some("RunMat:hann:ArgumentCount"),
    when: "More than two options are supplied.",
    message: "hann: too many input arguments",
};
pub const HANN_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.PROVIDER_FAILED",
    identifier: Some("RunMat:hann:ProviderFailed"),
    when: "A supported provider window operation fails during execution.",
    message: "hann: provider window generation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    HANN_ERROR_INVALID_LENGTH,
    HANN_ERROR_INVALID_OPTION,
    HANN_ERROR_UNKNOWN_OPTION,
    HANN_ERROR_INTERNAL,
    HANN_ERROR_ARG_COUNT,
    HANN_ERROR_PROVIDER_FAILED,
];
pub const HANN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
