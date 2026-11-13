use crate::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};

pub(super) const ISCATEGORICAL: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "iscategorical is a universal object-type predicate; integer host or resident values return scalar false without gathering or converting numeric payload data.",
};

pub(super) const ISORDINAL: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isordinal accepts any data type and returns true only for ordinal categorical arrays; integer host or resident values return scalar false without payload access.",
};

pub(super) const ISTABLE: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "istable is a universal object-type predicate; integer host or resident values return scalar false without gathering payload data.",
};

pub(super) const ISTIMETABLE: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "istimetable is a universal object-type predicate; integer host or resident values return scalar false without gathering payload data.",
};
