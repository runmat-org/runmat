use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nanmin"), slug: Some("nanmin"),
    summary: "Minimum that ignores NaN values.",
    description: "`nanmin(A, ...)` is the NaN-omitting compatibility alias for `min(A, ..., \"omitnan\")`.",
    keywords: &["nanmin", "min", "omitnan", "missing"],
    related: &["min"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and output",
            paragraphs: &[
                "Reduction forms use min with omitnan; pass [] before a numeric dimension. A reduction can also return one-based indices as its second output. The legacy two-numeric-input form compares pairwise, takes the non-NaN operand when exactly one is NaN, and supports one output.",
                "Typed integer reduction and compatible pairwise forms retain exact integer storage in RunMat mode.",
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
        id: "omit-nan", title: "Ignore NaN values", program: "m = nanmin([NaN 2 3])",
        display_output: Some("m = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(m, 2));" },
    },
        BuiltinExample {
        id: "pairwise", title: "Pairwise minima", program: "B = nanmin([NaN 4 3], [2 NaN 5])",
        display_output: Some("B = [2 4 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [2 4 3]));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "min", target: BuiltinDocumentationLinkTarget::Builtin("min"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "NaN-aware alias admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Owning min reduction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/min.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "NaN-aware alias and integer policy", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/nan.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
