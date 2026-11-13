use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text.",
    },
    BuiltinParamDescriptor {
        name: "delimiters",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Delimiter characters.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tok",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Primary output.",
    },
    BuiltinParamDescriptor {
        name: "rem",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Secondary output.",
    },
];
pub const STRTOK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTOK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strtok:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "strtok: too many output arguments",
};

pub const STRTOK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "[tok, rem] = strtok(text, delimiters)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[STRTOK_ERROR_TOO_MANY_OUTPUTS],
};
pub const STRTOK_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
 kind: BuiltinIntegerAuditKind::NotApplicable,
 canonical_builtin: None,
 notes: "strtok accepts text and rejects numeric and provider-resident text roles before provider access.",
};
