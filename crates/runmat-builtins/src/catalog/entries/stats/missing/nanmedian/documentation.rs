use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nanmedian"), slug: Some("nanmedian"),
    summary: "Median that ignores NaN values.",
    description: "`nanmedian(A, ...)` is the NaN-omitting compatibility alias for `median(A, ..., \"omitnan\")`.",
    keywords: &["nanmedian", "median", "omitnan", "missing"],
    related: &["median"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and output",
            paragraphs: &[
                "Delegates to median with omitnan. Supported dimension, vector-dimension, and all options are forwarded.",
                "Typed integer reduction preserves exact native storage on its supported host path.",
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
        id: "omit-nan", title: "Ignore NaN values", program: "m = nanmedian([1 NaN 5], 'all')",
        display_output: Some("m = 3"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(m, 3));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "NaN-aware alias admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Owning median reduction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/median.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "NaN-aware alias and integer policy", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/nan.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
