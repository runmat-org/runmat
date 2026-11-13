use crate::*;

const UNICODE2NATIVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "uint8 row vector containing encoded bytes.",
}];

const UNICODE2NATIVE_INPUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "unicodestr",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unicode text, provided as a string scalar or character vector.",
}];

const UNICODE2NATIVE_INPUT_TEXT_ENCODING: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "unicodestr",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unicode text, provided as a string scalar or character vector.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Character encoding name.",
    },
];

const UNICODE2NATIVE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "bytes = unicode2native(unicodestr)",
        inputs: &UNICODE2NATIVE_INPUT_TEXT,
        outputs: &UNICODE2NATIVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = unicode2native(unicodestr, encoding)",
        inputs: &UNICODE2NATIVE_INPUT_TEXT_ENCODING,
        outputs: &UNICODE2NATIVE_OUTPUT,
    },
];

pub const UNICODE2NATIVE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNICODE2NATIVE.INVALID_INPUT",
    identifier: Some("RunMat:unicode2native:InvalidInput"),
    when: "Input text is not a string scalar or character vector.",
    message: "unicode2native: input must be a string scalar or character vector",
};

pub const UNICODE2NATIVE_ERROR_INVALID_ENCODING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNICODE2NATIVE.INVALID_ENCODING",
    identifier: Some("RunMat:unicode2native:InvalidEncoding"),
    when: "Encoding name is unsupported or malformed.",
    message: "unicode2native: unsupported character encoding",
};

pub const UNICODE2NATIVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNICODE2NATIVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:unicode2native:InvalidArgument"),
    when: "Too many arguments were supplied.",
    message: "unicode2native: invalid argument",
};

pub const UNICODE2NATIVE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNICODE2NATIVE.INTERNAL",
    identifier: Some("RunMat:unicode2native:InternalError"),
    when: "Internal byte-vector tensor construction failed.",
    message: "unicode2native: internal error",
};

const UNICODE2NATIVE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    UNICODE2NATIVE_ERROR_INVALID_INPUT,
    UNICODE2NATIVE_ERROR_INVALID_ENCODING,
    UNICODE2NATIVE_ERROR_INVALID_ARGUMENT,
    UNICODE2NATIVE_ERROR_INTERNAL,
    UNICODE2NATIVE_ERROR_TOO_MANY_OUTPUTS,
];

pub const UNICODE2NATIVE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNICODE2NATIVE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:unicode2native:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "unicode2native: too many output arguments",
};

pub const UNICODE2NATIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNICODE2NATIVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNICODE2NATIVE_ERRORS,
};
