use crate::*;

pub(super) const ISKEYWORD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"iskeyword"####),
    slug: Some(r####"iskeyword"####),
    summary: r####"Test MATLAB reserved keywords or list all keywords."####,
    description: r####"`iskeyword(s)` returns true when `s` is a MATLAB reserved keyword. `iskeyword` with no inputs returns a cell column containing the full keyword list as character vectors."####,
    keywords: &[
        r####"iskeyword"####,
        r####"keyword"####,
        r####"identifier"####,
        r####"language"####,
        r####"MATLAB compatibility"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"Accepts a string scalar or character vector for the predicate form."####,
                r####"Rejects non-text inputs and nonscalar string arrays."####,
                r####"The keyword list is shared with `isvarname`, so reserved words such as `for`, `end`, `classdef`, and `parfor` cannot be valid variable names."####,
                r####"The result of `iskeyword(s)` is a host logical scalar. The result of `iskeyword` is a cell column of character row vectors."####,
                r####"GPU support: Host text/language predicate. gpuArray inputs are invalid because the input must be text."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[
                r####"Test a reserved keyword
tf = iskeyword('for')
Expected display: tf = logical(1)"####,
                r####"Test ordinary text
tf = iskeyword("plot")
Expected display: tf = logical(0)"####,
                r####"Return all keywords
k = iskeyword
Expected display: k = {'break'; 'case'; ...; 'while'}"####,
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "typed-basic",
            title: "test a reserved word",
            program: "tf = iskeyword('if')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(tf);",
            },
        },
        BuiltinExample {
            id: "numeric-rejection",
            title: "Reject numeric keyword operand",
            program: "iskeyword(42)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:iskeyword:InvalidText",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: r####"Does `iskeyword` accept string arrays?"####,
            answer: r####"Only string scalars are accepted. Multi-element string arrays are rejected so the predicate remains a scalar MATLAB-compatible query."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Is the comparison case-sensitive?"####,
            answer: r####"Yes. MATLAB keywords are lowercase language tokens, so `iskeyword('for')` is true while `iskeyword('For')` is false."####,
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: r####"isvarname"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./isvarname"####),
        },
        BuiltinDocumentationLink {
            label: r####"namelengthmax"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./namelengthmax"####),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/identifiers.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/identifiers.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/identifiers/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/identifiers/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Legacy test reference",
            location: r####"builtins::introspection::identifiers::tests"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
