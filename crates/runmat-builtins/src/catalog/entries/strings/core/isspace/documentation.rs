use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`isspace(text)` classifies Unicode whitespace in a host string scalar or character array. Character arrays retain their shape; a string scalar produces a logical row by character.",
        "A one-element string-array carrier is unwrapped and classified as a string scalar; a multi-element string array errors with `RunMat:isspace:InvalidStringArray`. This differs from `isletter`, which returns scalar false for every string-array carrier. Other nontext inputs, including integers and resident numeric values, return scalar false without provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Character classification is host-only in native and browser runtimes, with no GPU kernel or fusion form. Result construction or an invalid string array can throw; no callback, workspace, filesystem, network, randomness, or time effect occurs.",
        "One output is supported. Distributed resident-input execution is unsupported.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-example",
        title: "Find whitespace",
        program: "tf = isspace(\"a b\")",
        display_output: Some("tf = [false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "character-matrix",
        title: "Classify a character matrix",
        program: "tf = isspace(char(\"a \" , \" b\"))",
        display_output: Some("tf is a 2-by-2 logical matrix"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(tf), [2 2])); assert(isequal(tf, logical([0 1; 1 0])));",
        },
    },
    BuiltinExample {
        id: "numeric-is-not-text",
        title: "Do not reinterpret a numeric space code as text",
        program: "tf = isspace(uint16(32))",
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
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("isspace"), slug: Some("isspace"),
    summary: "Return true for whitespace characters in text.", description: "`isspace(text)` returns logical whitespace classification for supported text.",
    keywords: &["isspace", "strings", "core", "matlab", "compatibility"], related: &["string", "char", "isletter", "isstrprop"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isspace.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isspace.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "scalar and array behavior", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy sidecar title, summary, signature, example, related links and source link retained with source updated to its extracted runtime module. Its blanket GPU-gather note was corrected: resident numeric values return false without provider access."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
