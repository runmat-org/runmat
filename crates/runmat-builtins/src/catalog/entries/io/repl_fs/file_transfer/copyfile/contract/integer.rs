use crate::*;

pub const COPYFILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Source, destination, and force inputs are text; integer values reject before filesystem access.",
};
