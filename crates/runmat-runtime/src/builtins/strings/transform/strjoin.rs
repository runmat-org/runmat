use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const STRJOIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Joined string array.",
}];

const STRJOIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text array.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter inserted between row elements.",
    },
];

const STRJOIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strjoin(text, delimiter)",
    inputs: &STRJOIN_INPUTS,
    outputs: &STRJOIN_OUTPUT,
}];

pub const STRJOIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRJOIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "strjoin",
    descriptor(self::STRJOIN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::strjoin"
)]
pub async fn strjoin_builtin(a: Value, delim: Value) -> crate::BuiltinResult<Value> {
    crate::strjoin_rowwise(a, delim).await
}
