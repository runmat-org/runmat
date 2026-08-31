mod floating;
mod integer;

pub(super) use floating::{FLINTMAX_DOCUMENTATION, REALMAX_DOCUMENTATION, REALMIN_DOCUMENTATION};
pub(super) use integer::{INTMAX_DOCUMENTATION, INTMIN_DOCUMENTATION};

use crate::{
    BuiltinDocumentationEvidence, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinEvidenceKind, BuiltinEvidenceReference,
};

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits.rs",
    ),
}];

const FLOATING_VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Floating limit representation tests",
        location: "builtins::math::elementwise::numeric_limits::tests",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Static contract and distributed-placement tests",
        location: "catalog::tests::numeric_limit_contracts_infer_class_representation_and_distributed_like_outputs",
    },
];

const INTEGER_VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Exact integer limit representation tests",
        location: "builtins::math::elementwise::numeric_limits::tests",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Static contract and distributed-placement tests",
        location: "catalog::tests::numeric_limit_contracts_infer_class_representation_and_distributed_like_outputs",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::ProviderTest,
        label: "WGPU class, ownership, and wide-value test",
        location: "builtins::math::elementwise::numeric_limits::tests::integer_limit_like_preserves_wgpu_class_and_wide_value",
    },
];

pub(super) const FLOATING_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: FLOATING_VERIFICATION,
    notes: &[],
};

pub(super) const INTEGER_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: INTEGER_VERIFICATION,
    notes: &[],
};
