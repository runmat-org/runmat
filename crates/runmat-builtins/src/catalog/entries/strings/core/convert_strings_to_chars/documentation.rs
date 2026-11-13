use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Conversion", paragraphs: &[
        "`convertStringsToChars` converts each top-level string input independently. A string scalar becomes a character array; a string array with more than one element becomes a same-shape cell array of character arrays. Empty and missing scalar strings become 0-by-0 character arrays.",
        "Cells, structs, numerical values, and all other non-string inputs pass through unchanged. Use `convertContainedStringsToChars` when nested string values must be converted.",
    ] },
    BuiltinDocumentationSection { heading: "Outputs and placement", paragraphs: &[
        "Outputs correspond to input positions. If more outputs are requested than inputs, RunMat fills the extra outputs with scalar double zero. Unrequested outputs are discarded, but every input is still converted.",
        "Non-target resident values retain their provider handle and owner; no GPU kernel, gather, fusion, or callback is used. String conversion materializes host text and may fail allocation. The builtin runs in native and browser runtimes.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-multiple",
        title: "Convert two string scalars",
        program: "[a,b] = convertStringsToChars(\"run\", \"mat\")",
        display_output: Some("a and b are character rows"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(a, 'run')); assert(isequal(b, 'mat'));",
        },
    },
    BuiltinExample {
        id: "string-array",
        title: "Convert a string array to cells",
        program: "C = convertStringsToChars([\"one\" \"two\"])",
        display_output: Some("C is a 1-by-2 cell array of character rows"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(iscell(C)); assert(isequal(size(C), [1 2])); assert(isequal(C{1}, 'one'));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("convertStringsToChars"), slug: Some("convertStringsToChars"),
    summary: "Convert string scalars and arrays to character vectors.", description: "Convert top-level string values to character arrays while passing non-string values through.",
    keywords: &["convertStringsToChars", "strings", "core", "matlab", "compatibility"], related: &["string", "char", "convertContainedStringsToChars"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_strings_to_chars.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_strings_to_chars.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "conversion and passthrough", location: "builtins::strings::core::compat::tests" }], notes: &["All substantive sidecar behavior, signature, examples, related links, and source disposition reconciled. Padding beyond input count is recorded as current RunMat behavior, not claimed MATLAB parity."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
