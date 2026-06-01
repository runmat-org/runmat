use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const DEAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distributed output values.",
}];

const DEAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values to distribute.",
}];

const DEAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = deal(varargin)",
    inputs: &DEAL_INPUTS,
    outputs: &DEAL_OUTPUT,
}];

pub const DEAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "deal",
    descriptor(self::DEAL_DESCRIPTOR),
    builtin_path = "crate::builtins::common::deal"
)]
pub async fn deal_builtin_registered(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::deal_builtin(rest).await
}
