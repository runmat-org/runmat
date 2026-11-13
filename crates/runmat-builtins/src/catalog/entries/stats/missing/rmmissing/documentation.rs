use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rmmissing"),
    slug: Some("rmmissing"),
    summary: "Remove missing vector entries, matrix rows or columns, or table rows.",
    description: "`rmmissing(A)` removes missing entries from a vector and removes matrix or table rows containing missing data by default.",
    keywords: &["rmmissing", "missing", "NaN", "string", "table"],
    related: &["ismissing", "fillmissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Selection and outputs",
            paragraphs: &[
                r#"Vectors preserve row or column orientation. Use dimension 2 or the `"columns"` option to remove matrix columns instead of rows. Tables retain variable names and row names when rows are selected."#,
                "The optional second output is a logical mask of removed vector entries, rows, or columns. Integer arrays have no standard missing elements, so every integer class is an exact no-op with an all-false mask.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Resident values and compatibility",
            paragraphs: &[
                "A typed-integer dimension control is a RunMat-only extension, checked before provider work. Ordinary numeric, text, and table forms remain available in MATLAB compatibility mode.",
                "Resident numeric input is downloaded through its exact owning provider; the result and logical removal mask are restored independently with class, shape, owner, and alias validation. Unsupported automatic restoration may leave a host result.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "remove-missing-columns",
        title: "Remove missing columns",
        program: "B = rmmissing([1 NaN 3; 4 5 6], 2)",
        display_output: Some("B = [1 3; 4 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1 3; 4 6]));" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ismissing", target: BuiltinDocumentationLinkTarget::Builtin("ismissing") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Removal admission and restoration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Removal algorithms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/removal.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Shape, integer, output mask, owner, and admission behavior", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/removal.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-data calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
