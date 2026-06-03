use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const MAKE_ANON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle_text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Anonymous function text handle.",
}];

const MAKE_ANON_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Comma-separated parameter list.",
    },
    BuiltinParamDescriptor {
        name: "body",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Anonymous body expression text.",
    },
];

const MAKE_ANON_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "handle_text = make_anon(params, body)",
    inputs: &MAKE_ANON_INPUTS,
    outputs: &MAKE_ANON_OUTPUT,
}];

pub const MAKE_ANON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAKE_ANON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runtime_builtin(
    name = "make_anon",
    category = "introspection",
    summary = "Create internal anonymous-function textual handle representation.",
    keywords = "anonymous,function,handle,internal",
    descriptor(crate::builtins::introspection::make_anon::MAKE_ANON_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::make_anon"
)]
pub async fn make_anon_builtin(params: String, body: String) -> crate::BuiltinResult<Value> {
    crate::make_anon_builtin(params, body).await
}
