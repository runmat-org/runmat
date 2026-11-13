use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Structure-container identity",
        paragraphs: &[
            "`isstruct(A)` returns one logical scalar. It is true for scalar, nonscalar, or empty structure arrays and false for objects, cells, and all other value classes.",
            "The predicate checks the input container, not the types or values of its fields. A structure stored inside a cell does not make the cell itself a structure array.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and placement",
        paragraphs: &["The host result is determined from value or distributed-handle type metadata. Numeric and resident payloads are never downloaded or converted."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Identify a scalar structure",
        program: "S = struct('temperature', 21); tf = isstruct(S)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "cell",
        title: "A cell containing a structure is still a cell",
        program: "C = {struct('temperature', 21)}; tf = isstruct(C)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "An empty structure array retains its container identity",
        program: "S = struct([]); tf = isstruct(S)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
];

pub(super) const ISSTRUCT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isstruct"),
    slug: Some("isstruct"),
    summary: "Determine whether a value is a structure array.",
    description: "`isstruct` returns a logical scalar based on the input's structure-container identity.",
    keywords: &["isstruct", "struct", "structure array", "predicate"],
    related: &["struct", "iscell", "isfield"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB isstruct reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/isstruct.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Structure-container predicate runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isstruct.rs"),
        }],
        verification: &[],
        notes: &["Source-only author packet; executable examples and provider/distributed behavior await qualification."],
    },
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
