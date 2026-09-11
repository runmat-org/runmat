use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Cell arrays of character arrays",
        paragraphs: &[
            "`iscellstr(A)` returns true when `A` is an empty cell array or every cell contains a character array. Noncell values and cells containing any other value class return false.",
            "Character arrays of any size satisfy the predicate. This includes character matrices; the predicate is broader than APIs that require every element to be a character row vector.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Character arrays and string values",
        paragraphs: &[
            "A MATLAB string scalar or string array is a different value class from a character array. A cell containing string values therefore returns false.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "character-vectors",
        title: "Check character vectors in a cell array",
        program: "C = {'red', 'blue'};\ntf = iscellstr(C)",
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
        id: "character-matrix",
        title: "Accept a character matrix inside a cell",
        program: "C = {['ab'; 'cd']};\ntf = iscellstr(C)",
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
        title: "Accept an empty cell array",
        program: "C = cell(0, 0);\ntf = iscellstr(C)",
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
        id: "mixed-cell",
        title: "Reject a cell with a noncharacter member",
        program: "C = {'red', 7};\ntf = iscellstr(C)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "string-array",
        title: "Distinguish string arrays from cell arrays of characters",
        program: "S = [\"red\" \"blue\"];\ntf = iscellstr(S)",
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
        question: "Do character matrices satisfy `iscellstr`?",
        answer: "Yes. Every cell must contain a character array, but that array need not be a row vector.",
    },
    BuiltinDocumentationFaq {
        question: "Do string scalars satisfy `iscellstr`?",
        answer: "No. String values and character arrays are distinct classes.",
    },
    BuiltinDocumentationFaq {
        question: "Why does an empty cell array return true?",
        answer: "It contains no element that violates the character-array requirement.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "iscell",
        target: BuiltinDocumentationLinkTarget::Builtin("iscell"),
    },
    BuiltinDocumentationLink {
        label: "Compatible iscellstr reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/iscellstr.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Cell-character predicate runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/iscellstr.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Empty, character-vector, character-matrix, string, and mixed-cell behavior",
        location: "crates/runmat-runtime/src/builtins/logical/tests/iscellstr/tests.rs",
    }],
    notes: &[],
};

pub(super) const ISCELLSTR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("iscellstr"),
    slug: Some("iscellstr"),
    summary: "Determine whether a value is a cell array of character arrays.",
    description: "`iscellstr` returns one logical scalar after checking the input container and, for cell arrays, every member's value class.",
    keywords: &["iscellstr", "cellstr", "cell", "character", "predicate"],
    related: &["iscell"],
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
