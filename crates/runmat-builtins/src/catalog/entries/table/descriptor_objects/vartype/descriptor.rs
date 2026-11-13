use crate::*;

pub const VARTYPE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "vartype accepts one host string scalar or character vector naming a variable class; integer and resident numeric values reject before provider access or selector construction.",
};

pub const VARTYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "S = vartype(type)",
        inputs: &[BuiltinParamDescriptor {
            name: "type",
            ty: BuiltinParamType::StringScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Host string scalar or character row naming a variable class.",
        }],
        outputs: &[BuiltinParamDescriptor {
            name: "descriptor",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Scalar host value descriptor; not a live iterator or UI handle.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
