use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Container identity",
        paragraphs: &[
            "`iscell(A)` returns one logical scalar. It is true when `A` is a cell array and false for every other supported value class.",
            "The predicate does not inspect the elements of a cell array. Empty, nested, and heterogeneous cell arrays all return true.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and placement",
        paragraphs: &[
            "The result is determined from the value's container identity. Numeric payloads are not converted or downloaded, and the logical result is produced on the host.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "cell",
        title: "Detect a cell array",
        program: "A = {1, 'runmat'};\ntf = iscell(A)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "empty-cell",
        title: "Detect an empty cell array",
        program: "A = cell(0, 2);\ntf = iscell(A)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "nested-cell",
        title: "Check a nested cell array without inspecting its contents",
        program: "A = {{1, 2}, [3 4]};\ntf = iscell(A)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "numeric",
        title: "Reject a numeric array",
        program: "A = [1 2 3];\ntf = iscell(A)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `iscell` inspect the values stored in a cell array?",
        answer: "No. It checks only whether the input itself is a cell array.",
    },
    BuiltinDocumentationFaq {
        question: "Does an empty cell array return true?",
        answer: "Yes. Empty cell arrays retain their cell container identity.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "iscellstr",
        target: BuiltinDocumentationLinkTarget::Builtin("iscellstr"),
    },
    BuiltinDocumentationLink {
        label: "Compatible iscell reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/iscell.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Cell-container predicate runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/iscell.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Cell, empty, nested, numeric, and output-count behavior",
        location: "crates/runmat-runtime/src/builtins/logical/tests/iscell/tests.rs",
    }],
    notes: &[],
};

pub(super) const ISCELL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("iscell"),
    slug: Some("iscell"),
    summary: "Determine whether a value is a cell array.",
    description: "`iscell` returns one logical scalar based on the input's container identity.",
    keywords: &["iscell", "cell", "container", "predicate"],
    related: &["iscellstr"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
