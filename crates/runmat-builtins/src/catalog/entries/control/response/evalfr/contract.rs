use crate::*;

const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO tf model.",
};
const POINT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite scalar point in the continuous s plane or discrete z plane.",
};
const RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "frsp",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex double response scalar.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [SYS, POINT];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [RESPONSE];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frsp = evalfr(sys,x)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.EVALFR.INVALID_ARGUMENT",
        identifier: Some("RunMat:evalfr:InvalidArgument"),
        when: "Point or invocation is invalid, singular, or cannot produce a finite response.",
        message: "evalfr: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.EVALFR.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:evalfr:UnsupportedModel"),
        when: "The model is outside the supported zero-delay SISO tf domain.",
        message: "evalfr: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.EVALFR.INVALID_MODEL",
        identifier: Some("RunMat:evalfr:InvalidModel"),
        when: "A required transfer-function property is malformed.",
        message: "evalfr: invalid model",
    },
];
pub const EVALFR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
