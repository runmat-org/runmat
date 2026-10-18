use crate::*;

const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO tf model.",
};
const W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Finite nonnegative host angular-frequency vector.",
};
const H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex double response array of size 1×1×N.",
};
const WOUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "wout",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular-frequency column vector.",
};
const SYS_INPUT: [BuiltinParamDescriptor; 1] = [SYS];
const SYS_W_INPUT: [BuiltinParamDescriptor; 2] = [SYS, W];
const H_OUTPUT: [BuiltinParamDescriptor; 1] = [H];
const H_W_OUTPUT: [BuiltinParamDescriptor; 2] = [H, WOUT];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "H = freqresp(sys)",
        inputs: &SYS_INPUT,
        outputs: &H_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqresp(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &H_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[H,wout] = freqresp(sys)",
        inputs: &SYS_INPUT,
        outputs: &H_W_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[H,wout] = freqresp(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &H_W_OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.FREQRESP.INVALID_ARGUMENT",
        identifier: Some("RunMat:freqresp:InvalidArgument"),
        when: "Invocation, frequency vector, or evaluation point is invalid or singular.",
        message: "freqresp: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.FREQRESP.INVALID_MODEL",
        identifier: Some("RunMat:freqresp:InvalidModel"),
        when: "A required transfer-function property is malformed.",
        message: "freqresp: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.FREQRESP.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:freqresp:UnsupportedModel"),
        when: "The model is outside the supported zero-delay SISO tf domain.",
        message: "freqresp: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.FREQRESP.INTERNAL",
        identifier: Some("RunMat:freqresp:Internal"),
        when: "Response tensor materialization fails.",
        message: "freqresp: internal error",
    },
];
pub const FREQRESP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
