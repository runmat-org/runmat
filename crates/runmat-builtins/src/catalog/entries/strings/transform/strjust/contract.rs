use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Justified text with the input container shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String, character, or cell text.",
    },
    BuiltinParamDescriptor {
        name: "side",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("right"),
        description: "Scalar string or character alignment; left, right (default), or center.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "s = strjust(text, side)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRJUST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const STRJUST_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "strjust operates on character, string, or cell text and an optional textual alignment. Numeric, integer and provider-resident inputs reject before provider access; integers are not treated as character codes." };
