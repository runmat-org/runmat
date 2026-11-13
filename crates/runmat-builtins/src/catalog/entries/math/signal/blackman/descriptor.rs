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
        label: "w = blackman(n)",
        inputs: &INPUT_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, sampling)",
        inputs: &INPUT_TWO_SAMPLING,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, precision)",
        inputs: &INPUT_TWO_PRECISION,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, sampling, precision)",
        inputs: &INPUT_THREE,
        outputs: &OUTPUT,
    },
];
pub const BLACKMAN_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INVALID_LENGTH",
    identifier: Some("RunMat:blackman:InvalidLength"),
    when: "Length is not a supported nonnegative scalar.",
    message: "blackman: expected a nonnegative scalar length",
};
pub const BLACKMAN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INVALID_OPTION",
    identifier: Some("RunMat:blackman:InvalidOption"),
    when: "An option is not a scalar text token.",
    message: "blackman: unrecognized option",
};
pub const BLACKMAN_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.UNKNOWN_OPTION",
    identifier: Some("RunMat:blackman:UnknownOption"),
    when: "An option is unrecognized or out of position.",
    message: "blackman: unrecognized option",
};
pub const BLACKMAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INTERNAL",
    identifier: Some("RunMat:blackman:InternalError"),
    when: "Window materialization fails.",
    message: "blackman: internal error",
};
pub const BLACKMAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.ARG_COUNT",
    identifier: Some("RunMat:blackman:ArgumentCount"),
    when: "More than two options are supplied.",
    message: "blackman: too many input arguments",
};
pub const BLACKMAN_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.PROVIDER_FAILED",
    identifier: Some("RunMat:blackman:ProviderFailed"),
    when: "A supported provider window operation fails during execution.",
    message: "blackman: provider window generation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    BLACKMAN_ERROR_INVALID_LENGTH,
    BLACKMAN_ERROR_INVALID_OPTION,
    BLACKMAN_ERROR_UNKNOWN_OPTION,
    BLACKMAN_ERROR_INTERNAL,
    BLACKMAN_ERROR_ARG_COUNT,
    BLACKMAN_ERROR_PROVIDER_FAILED,
];
pub const BLACKMAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
