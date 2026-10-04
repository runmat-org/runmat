//! Receiver-family integer applicability; payload controls have separate form capabilities.

use crate::*;

pub(super) const DATASET_NONATTRIBUTE_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Dataset identity, array-name lookup/listing, membership, begin, snapshot, and refresh accept a managed Dataset receiver plus text controls; these forms do not consume integer values or indices or return an integer-class payload.",
    };
