use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nanvar"), slug: Some("nanvar"),
    summary: "Variance that ignores NaN values.",
    description: "`nanvar(A, ...)` is the NaN-omitting compatibility alias for `var(A, ..., \"omitnan\")`.",
    keywords: &["nanvar", "var", "omitnan", "missing"],
    related: &["var"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and output",
            paragraphs: &[
                "Delegates to var with omitnan. Normalization, dimension, and all controls follow the admitted parent reduction contract.",
                "Typed integer data is unsupported; typed integer normalization or dimension controls are a RunMat-only extension.",
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
        id: "omit-nan", title: "Ignore NaN values", program: "v = nanvar([1 NaN 3], 0, 'all')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(abs(v - 2) < 1e-12);" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "var", target: BuiltinDocumentationLinkTarget::Builtin("var"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "NaN-aware alias admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Owning var reduction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/var.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "NaN-aware alias and integer policy", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/nan.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
