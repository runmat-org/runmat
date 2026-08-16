use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, Value,
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

pub const STRJOIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strjoin joins text using a text delimiter. Integer, numeric, and provider-resident values have no input role and reject before provider access without implicit text conversion.",
};

#[runmat_macros::runtime_builtin(
    name = "strjoin",
    descriptor(self::STRJOIN_DESCRIPTOR),
    integer_audit(self::STRJOIN_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::strjoin"
)]
pub async fn strjoin_builtin(a: Value, delim: Value) -> crate::BuiltinResult<Value> {
    crate::strjoin_rowwise(a, delim).await
}
