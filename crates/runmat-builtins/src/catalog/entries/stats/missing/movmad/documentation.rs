use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("movmad"),
    slug: Some("movmad"),
    summary: "Compute centered moving median absolute deviations.",
    description: "`movmad(A, k)` computes median absolute deviation in a centered window of length `k`.",
    keywords: &["movmad", "moving", "median", "absolute deviation", "missing"],
    related: &["nanmedian", "median"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Window and missing-value policy",
            paragraphs: &[
                r#"Real numeric vectors and matrices are supported. The first non-singleton dimension is the default; pass dimension 1 or 2 directly, or as `"dim", value`."#,
                r#"`"omitnan"` ignores NaNs within a window; `"includenan"` propagates them. Integer data is accepted but the deviation output is double; single and double floating inputs retain their respective precision."#,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Resident input",
            paragraphs: &[
                "Resident input is gathered through its exact owning provider for the host calculation. A representable result is restored through that owner, with output metadata and alias validation; this is not a GPU moving-window kernel.",
                "A resident window longer than 31 is a RunMat-only extension. In MATLAB compatibility mode that form is rejected before provider access.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "moving-column",
        title: "Moving MAD over a column",
        program: "M = movmad([1; 2; 100; 4; 5], 3)",
        display_output: Some("M = [0.5; 1; 2; 1; 0.5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(M, [0.5; 1; 2; 1; 0.5]));" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "nanmedian", target: BuiltinDocumentationLinkTarget::Builtin("nanmedian") },
        BuiltinDocumentationLink { label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Admission and owner-aware restoration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Moving MAD algorithm", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/moving.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Window, numeric class, compatibility, and provider-admission tests", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/moving.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-data calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
