use crate::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};

pub const ISFIELD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isfield is a structure-metadata predicate. Integer targets return false, while integer field-name arguments reject without conversion or provider access.",
};
