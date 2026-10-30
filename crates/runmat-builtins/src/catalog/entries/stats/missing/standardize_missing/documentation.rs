use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("standardizeMissing"),
    slug: Some("standardizeMissing"),
    summary: "Replace user-specified missing indicators with canonical missing values.",
    description: "`standardizeMissing(A, indicators)` changes matching floating entries to NaN and matching string elements to typed missing values, displayed as `<missing>`. Supported cell and table values are processed recursively.",
    keywords: &["standardizeMissing", "missing", "NaN", "string", "table"],
    related: &["ismissing", "fillmissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Indicators and classes",
            paragraphs: &[
                "Numeric arrays accept scalar or array indicators; a NaN indicator matches existing NaNs. Text arrays and character-array rows accept text indicators. Character arrays become missing-aware string arrays, with one output entry per row.",
                "All eight fixed-width integer classes can serve as indicators for supported floating data. Comparison uses the integer's exact stored value: an integer that binary64 cannot represent cannot falsely match a rounded double. Integer variables inside tables retain their native values.",
                "A direct native-integer data array is a RunMat-only exact no-op because its class has no standard missing element. MATLAB-compatible mode rejects that form before gathering resident input. An explicitly resident indicator is separately gated; automatically resident inputs gather through their owners for host standardization.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "numeric-sentinel",
            title: "Convert a numeric sentinel to NaN",
            program: "B = standardizeMissing([-99 2 4], -99)",
            display_output: Some("B = [NaN 2 4]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isnan(B(1))); assert(isequal(B(2:3), [2 4]));",
            },
        },
        BuiltinExample {
            id: "exact-integer-indicator",
            title: "Avoid a false wide-integer match",
            program: "B = standardizeMissing([9007199254740992 1], uint64(9007199254740992) + uint64(1))",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(B, [9007199254740992 1]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ismissing", target: BuiltinDocumentationLinkTarget::Builtin("ismissing") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/standardize.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Typed indicator matching", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/standardize.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, integer, text, and compatibility behavior", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/standardize.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled sentinel standardization", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
