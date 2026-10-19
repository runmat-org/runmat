use crate::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};

pub const FIELDNAMES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fieldnames inspects structure or object metadata. Numeric values, including fixed-width integers and resident arrays, are not applicable inputs and reject without conversion or provider access.",
    };
