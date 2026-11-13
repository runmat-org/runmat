use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded numeric, logical, character, string, struct, or cell value.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "JSON text as a character row, string scalar, or one-element string array.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = jsondecode(text)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];

pub const JSONDECODE_ERROR_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONDECODE.INPUT_TYPE",
    identifier: None,
    when: "Input text is not a character vector or string scalar.",
    message: "jsondecode: JSON text must be a character vector or string scalar",
};
pub const JSONDECODE_ERROR_PARSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONDECODE.PARSE",
    identifier: None,
    when: "Input text is not valid JSON.",
    message: "jsondecode: invalid JSON text",
};
pub const JSONDECODE_ERROR_NUMERIC_LITERAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONDECODE.NUMERIC_LITERAL",
    identifier: None,
    when: "A JSON numeric literal cannot be represented as a finite RunMat double.",
    message: "jsondecode: invalid JSON numeric literal",
};
pub const JSONDECODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONDECODE.INTERNAL",
    identifier: None,
    when: "Internal conversion from parsed JSON to RunMat values fails.",
    message: "jsondecode: internal conversion failed",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    JSONDECODE_ERROR_INPUT_TYPE,
    JSONDECODE_ERROR_PARSE,
    JSONDECODE_ERROR_NUMERIC_LITERAL,
    JSONDECODE_ERROR_INTERNAL,
];
pub const JSONDECODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const JSONDECODE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "jsondecode accepts JSON text only. Integer and resident numeric inputs reject before provider access; JSON numeric literals decode to doubles.",
};
