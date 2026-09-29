use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Callable,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle invoked once for each matching parameter leaf.",
    },
    BuiltinParamDescriptor {
        name: "params, A1, ...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "One or more matching scalar, cell, structure, learnables table, or dlnetwork trees.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One reconstructed tree for each output of the callback.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[params, X1, ...] = dlupdate(fun, params, A1, ...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const DLUPDATE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "A function handle or parameter tree is missing, container shapes/fields/learnables metadata differ, traversal exceeds the supported depth, or a callback returns the wrong number of outputs.",
    message: "dlupdate: invalid callback, parameter trees, or output count",
};
pub const DLUPDATE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "An input object has no supported learnables, tabular, dlnetwork, or dlarray traversal contract.",
    message: "dlupdate: unsupported object tree",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [DLUPDATE_INVALID_INPUT, DLUPDATE_UNSUPPORTED];
pub const DLUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
