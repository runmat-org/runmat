use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nanmean"), slug: Some("nanmean"),
    summary: "Mean that ignores NaN values.",
    description: "`nanmean(A, ...)` is the NaN-omitting compatibility alias for `mean(A, ..., \"omitnan\")`.",
    keywords: &["nanmean", "mean", "omitnan", "missing"],
    related: &["mean", "fillmissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and output",
            paragraphs: &[
                "Delegates to mean with omitnan. Dimension, vector-dimension, all, native, and like forms follow the admitted parent reduction contract.",
                "Default integer input returns double; native output preserves the input integer class.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution",
            paragraphs: &[
                "The alias uses the admitted parent reduction implementation, including its supported resident/provider route and structured fallback. A successful catalog binding alone does not demonstrate a GPU kernel for every option or class.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "omit-nan", title: "Ignore NaN values", program: "m = nanmean([1 NaN 3], 'all')",
        display_output: Some("m = 2"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(abs(m - 2) < 1e-12);" },
    }],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean"),
        },
        BuiltinDocumentationLink {
            label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "NaN-aware alias admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Owning mean reduction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/mean.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "NaN-aware alias and integer policy", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/nan.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
