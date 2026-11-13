use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
}];
const EVALUATION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text.",
    },
    BuiltinParamDescriptor {
        name: "Evaluation",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the evaluation method option.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "'all' (default) or 'restricted'.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Primary output.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Secondary output.",
    },
];
pub const STR2NUM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2NUM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:str2num:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "str2num: too many output arguments",
};

pub const STR2NUM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[x, tf] = str2num(text)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x, tf] = str2num(text, Evaluation=method)",
            inputs: &EVALUATION_INPUTS,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[STR2NUM_ERROR_TOO_MANY_OUTPUTS],
};
pub const STR2NUM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
 kind: BuiltinIntegerAuditKind::NotApplicable,
 canonical_builtin: None,
 notes: "str2num accepts host text; integer/numeric and provider-resident input rejects before provider access. Numeric output is a separate result-class concern.",
};
