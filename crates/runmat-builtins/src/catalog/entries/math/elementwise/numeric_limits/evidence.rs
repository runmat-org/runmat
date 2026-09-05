use crate::{
    BuiltinDocumentationEvidence, BuiltinDocumentationFaq, BuiltinDocumentationLink,
    BuiltinDocumentationLinkTarget, BuiltinEvidenceKind, BuiltinEvidenceReference,
};

pub(super) const FLOATING_GPU_PARAGRAPHS: &[&str] = &[
    "Class-name forms return host scalars. With a floating `gpuArray` prototype, the `like` form creates only the scalar result on the prototype's registered provider and device. It preserves precision, complexity, storage kind, and explicit-placement provenance without downloading the prototype.",
    "Numeric-limit queries are scalar construction boundaries rather than elementwise or reduction kernels, so they are not fused with neighboring operations.",
];

pub(super) const LIKE_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does the `like` form copy the prototype shape?",
    answer: "No. It returns a 1-by-1 value while preserving class, complexity, sparsity, and applicable placement.",
};

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits",
    ),
}];

const STATIC_VERIFICATION: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::IntegrationTest,
    label: "Static contract and distributed-placement tests",
    location: "catalog::inference::math::numeric_limits::tests",
};

const FLOATING_VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Floating limit representation tests",
        location: "builtins::math::elementwise::numeric_limits::tests::floating",
    },
    STATIC_VERIFICATION,
];

const INTEGER_VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Exact integer limit representation tests",
        location: "builtins::math::elementwise::numeric_limits::tests::integer",
    },
    STATIC_VERIFICATION,
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::ProviderTest,
        label: "WGPU class, ownership, and wide-value test",
        location: "builtins::math::elementwise::numeric_limits::tests::provider::integer_like_preserves_wgpu_class_and_wide_value",
    },
];

pub(super) const FLOATING: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: FLOATING_VERIFICATION,
    notes: &[],
};

pub(super) const INTEGER: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: INTEGER_VERIFICATION,
    notes: &[],
};
