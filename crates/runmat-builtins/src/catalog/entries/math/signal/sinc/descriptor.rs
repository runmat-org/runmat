use crate::*;
const SINC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise normalized sinc output.",
}];

const SINC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input values.",
}];

const SINC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sinc(X)",
    inputs: &SINC_INPUTS,
    outputs: &SINC_OUTPUT,
}];

pub const SINC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.INVALID_INPUT",
    identifier: Some("RunMat:sinc:InvalidInput"),
    when: "Input cannot be interpreted as numeric/complex data.",
    message: "sinc: expected numeric input",
};

pub const SINC_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.PROVIDER_FAILED",
    identifier: Some("RunMat:sinc:ProviderFailed"),
    when: "Provider unary_sinc dispatch fails with a non-unsupported error.",
    message: "sinc: GPU provider unary_sinc failed",
};

pub const SINC_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.GATHER_FAILED",
    identifier: Some("RunMat:gather:DownloadFailed"),
    when: "GPU host fallback cannot download source data.",
    message: "gather: download failed",
};

pub const SINC_ERROR_MALFORMED_COMPLEX_BUFFER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.MALFORMED_COMPLEX_BUFFER",
    identifier: Some("RunMat:sinc:MalformedComplexBuffer"),
    when: "Complex-interleaved fallback buffer has odd element count.",
    message: "sinc: malformed complex buffer, odd length",
};

pub const SINC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.INTERNAL",
    identifier: Some("RunMat:sinc:InternalError"),
    when: "Internal tensor conversion/materialization fails.",
    message: "sinc: internal error",
};

const SINC_ERRORS: [BuiltinErrorDescriptor; 5] = [
    SINC_ERROR_INVALID_INPUT,
    SINC_ERROR_PROVIDER_FAILED,
    SINC_ERROR_GATHER_FAILED,
    SINC_ERROR_MALFORMED_COMPLEX_BUFFER,
    SINC_ERROR_INTERNAL,
];

pub const SINC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SINC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SINC_ERRORS,
};
