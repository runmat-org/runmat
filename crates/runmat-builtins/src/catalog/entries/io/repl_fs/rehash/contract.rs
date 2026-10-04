use crate::*;

const MODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "path, toolbox, or toolboxcache.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat status extension: zero when the source cache refresh completes.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "rehash",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "rehash(mode)",
        inputs: &[MODE],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = rehash(mode)",
        inputs: &[MODE],
        outputs: &[STATUS],
    },
    BuiltinSignatureDescriptor {
        label: "status = rehash()",
        inputs: &[],
        outputs: &[STATUS],
    },
];
const TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rehash:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "rehash: too many output arguments",
};
pub const REHASH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[TOO_MANY_OUTPUTS],
};
pub const REHASH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "rehash accepts only a host text mode; it does not compute on integer values.",
};
