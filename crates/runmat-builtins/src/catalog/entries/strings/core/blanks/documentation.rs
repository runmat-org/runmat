use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`blanks(n)` creates a one-row character array containing n spaces. Scalar real double, single, and all eight fixed-width integer classes are accepted when integral; zero and negative values produce an empty character row.",
        "Native integer lengths are read exactly, without a floating conversion. Nonintegral, nonfinite, non-scalar, or unrepresentable lengths reject. Very large valid lengths can fail host allocation.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and compatibility", paragraphs: &[
        "The character array is allocated on the host in native and browser runtimes. A resident numeric length is a RunMat-only extension: strict MATLAB compatibility rejects it before provider access; RunMat mode gathers it and may suspend. There is no GPU character kernel or fusion form.",
        "The function has no callback, workspace, filesystem, network, randomness, or time effect. At most one output is supported.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-example",
        title: "Create a row of spaces",
        program: "s = blanks(3)",
        display_output: Some("s is a 1-by-3 character row"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(s), [1 3])); assert(isequal(s, '   '));",
        },
    },
    BuiltinExample {
        id: "negative-count",
        title: "Clamp a negative count to zero",
        program: "s = blanks(int16(-2))",
        display_output: Some("s is an empty character row"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(s)); assert(isequal(size(s), [1 0]));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("blanks"), slug: Some("blanks"),
    summary: "Return a character row vector of spaces.", description: "`blanks(n)` returns a 1-by-n character row containing space characters.",
    keywords: &["blanks", "strings", "core", "matlab", "compatibility"], related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does an integer length produce an integer array?", answer: "No. The integer controls the output width; the result is a character row." }],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/blanks.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/blanks.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integer and compatibility behavior", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy description, four behaviors, example, FAQ, two related links and source reconciled. Resident-length behavior is made precise for strict and RunMat modes."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
