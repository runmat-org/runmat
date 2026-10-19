use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nansum"), slug: Some("nansum"),
    summary: "Sum that ignores NaN values.",
    description: "`nansum(A, ...)` is the NaN-omitting compatibility alias for `sum(A, ..., \"omitnan\")`.",
    keywords: &["nansum", "sum", "omitnan", "missing"],
    related: &["sum", "nanmean"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and output",
            paragraphs: &[
                "Delegates to sum with omitnan. Supported dimensions, all, and output-class options are forwarded after missing-policy admission.",
                "Default integer input returns double; native output preserves the integer class with saturating accumulation.",
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
        id: "omit-nan", title: "Ignore NaN values", program: "s = nansum([1 NaN 3], 'all')",
        display_output: Some("s = 4"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(s, 4));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum"),
        },
        BuiltinDocumentationLink {
            label: "nanmean", target: BuiltinDocumentationLinkTarget::Builtin("nanmean"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "NaN-aware alias admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Owning sum reduction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/sum.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "NaN-aware alias and integer policy", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/nan.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
