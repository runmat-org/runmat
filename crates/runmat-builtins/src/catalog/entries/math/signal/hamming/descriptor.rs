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
        label: "w = hamming(n)",
        inputs: &INPUT_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, sampling)",
        inputs: &INPUT_TWO_SAMPLING,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, precision)",
        inputs: &INPUT_TWO_PRECISION,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, sampling, precision)",
        inputs: &INPUT_THREE,
        outputs: &OUTPUT,
    },
];
pub const HAMMING_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INVALID_LENGTH",
    identifier: Some("RunMat:hamming:InvalidLength"),
    when: "Length is not a supported nonnegative scalar.",
    message: "hamming: expected a nonnegative scalar integer length",
};
pub const HAMMING_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INVALID_OPTION",
    identifier: Some("RunMat:hamming:InvalidOption"),
    when: "An option is not a scalar text token.",
    message: "hamming: unrecognized option",
};
pub const HAMMING_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.UNKNOWN_OPTION",
    identifier: Some("RunMat:hamming:UnknownOption"),
    when: "An option is unrecognized or out of position.",
    message: "hamming: unrecognized option",
};
pub const HAMMING_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INTERNAL",
    identifier: Some("RunMat:hamming:InternalError"),
    when: "Window materialization fails.",
    message: "hamming: internal error",
};
pub const HAMMING_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.ARG_COUNT",
    identifier: Some("RunMat:hamming:ArgumentCount"),
    when: "More than two options are supplied.",
    message: "hamming: too many input arguments",
};
pub const HAMMING_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.PROVIDER_FAILED",
    identifier: Some("RunMat:hamming:ProviderFailed"),
    when: "A supported provider window operation fails during execution.",
    message: "hamming: provider window generation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    HAMMING_ERROR_INVALID_LENGTH,
    HAMMING_ERROR_INVALID_OPTION,
    HAMMING_ERROR_UNKNOWN_OPTION,
    HAMMING_ERROR_INTERNAL,
    HAMMING_ERROR_ARG_COUNT,
    HAMMING_ERROR_PROVIDER_FAILED,
];
pub const HAMMING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
