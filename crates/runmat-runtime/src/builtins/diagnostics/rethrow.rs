use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const RETHROW_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const RETHROW_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error value to rethrow.",
}];

const RETHROW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "rethrow(err)",
    inputs: &RETHROW_INPUTS,
    outputs: &RETHROW_OUTPUT,
}];

pub const RETHROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RETHROW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "rethrow",
    descriptor(self::RETHROW_DESCRIPTOR),
    builtin_path = "crate::builtins::diagnostics::rethrow"
)]
pub async fn rethrow_builtin_registered(e: Value) -> crate::BuiltinResult<Value> {
    crate::rethrow_builtin(e).await
}
