use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`isletter(text)` classifies each Unicode scalar value in a host string scalar or character array and returns logical values in the text shape. A character matrix retains its shape.",
        "Other input types, including every string-array carrier (even one with a single element), integers, and provider-resident numeric values, return scalar false before provider access. A string scalar is distinct from a one-element string-array carrier in this RunMat implementation. No numeric-to-character conversion occurs.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Character classification runs on the host in native and browser runtimes. There is no GPU kernel or fusion form; no callback, workspace, filesystem, network, randomness, or time effect occurs.",
        "One output is supported. Result construction can throw; nontext input is a scalar-false route rather than an error.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-example",
        title: "Classify letters",
        program: "tf = isletter(\"a1\")",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "character-matrix",
        title: "Preserve a character matrix shape",
        program: "tf = isletter(char(\"a1\", \"2B\"))",
        display_output: Some("tf is a 2-by-2 logical matrix"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(tf), [2 2])); assert(isequal(tf, logical([1 0; 0 1])));",
        },
    },
    BuiltinExample {
        id: "numeric-is-not-text",
        title: "Do not reinterpret a numeric character code as text",
        program: "tf = isletter(uint16(65))",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, false));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("isletter"), slug: Some("isletter"),
    summary: "Return true for letters in text.", description: "`isletter(text)` classifies Unicode letters in a string scalar or character array.",
    keywords: &["isletter", "strings", "core", "matlab", "compatibility"], related: &["string", "char", "isspace", "isstrprop"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isletter.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isletter.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "character and numeric behavior", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy sidecar title, summary, signature, example, related links and source link retained with source updated to its extracted runtime module. Its GPU-gather note was corrected: resident numeric values return false without a provider call."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
