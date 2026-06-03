use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const FEVAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Function return value(s).",
}];

const FEVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle, handle text, closure, or object receiver.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Function call arguments.",
    },
];

const FEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = feval(f, varargin)",
    inputs: &FEVAL_INPUTS,
    outputs: &FEVAL_OUTPUT,
}];

pub const FEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &crate::FEVAL_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "feval",
    descriptor(self::FEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::feval"
)]
pub async fn feval_builtin_registered(f: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::feval_builtin(f, rest).await
}
