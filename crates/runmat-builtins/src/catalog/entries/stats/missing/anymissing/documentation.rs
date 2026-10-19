use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "nan-array",
        title: "Test for any missing value",
        program: "tf = anymissing([1 NaN 3])",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "integer-array",
        title: "Integer arrays have no standard missing value",
        program: "tf = anymissing(uint16([1 2 3]))",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, false));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("anymissing"),
    slug: Some("anymissing"),
    summary: "Return true when an input contains at least one missing value.",
    description: "`anymissing(A)` is the scalar predicate companion to `ismissing`. It recursively tests supported numeric, text, cell, structure, table, datetime, and duration values.",
    keywords: &["anymissing", "missing", "NaN", "table"],
    related: &["ismissing", "missing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Missing values",
            paragraphs: &[
                "Numeric NaN, missing strings, missing cells, structures with missing fields, and tables with a missing variable entry produce true. Fixed-width integer classes have no standard missing representation and produce false without floating conversion.",
                "The result is a host logical scalar. Resident input is gathered through its owning provider before recursive inspection; this builtin does not claim a resident reduction or fusion kernel.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ismissing", target: BuiltinDocumentationLinkTarget::Builtin("ismissing") },
        BuiltinDocumentationLink { label: "missing", target: BuiltinDocumentationLinkTarget::Builtin("missing") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Missing-value detection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/detection.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Nested and numeric missing-value semantics", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/core.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-value calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
