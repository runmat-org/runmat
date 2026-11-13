use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Conversion", paragraphs: &[
        "`convertContainedStringsToChars` converts top-level string values and strings nested at any cell or struct depth. String arrays with multiple elements become same-shape cells of character arrays. Cells and structs keep their structure and non-string members unchanged, including exact integer values and resident handles.",
        "Empty and missing string scalars convert to 0-by-0 character arrays. Typed struct arrays retain their dimensions and field identities.",
    ] },
    BuiltinDocumentationSection { heading: "Outputs and placement", paragraphs: &[
        "Multiple inputs convert independently. If more outputs are requested than inputs, RunMat fills the extra outputs with scalar double zero. Unrequested outputs are discarded, but every input is still converted.",
        "Conversion materializes host text and aggregate shells. Non-string resident members retain their original provider owner and are not gathered. There is no GPU kernel, fusion, callback, workspace, filesystem, network, randomness, or time effect. The builtin runs in native and browser runtimes.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "legacy-contained-cell",
    title: "Convert strings inside a cell",
    program: "C = {\"run\", uint64(7)}; D = convertContainedStringsToChars(C)",
    display_output: Some(
        "The string member becomes a character row; the integer member stays uint64",
    ),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(D{1}, 'run')); assert(isequal(D{2}, uint64(7)));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("convertContainedStringsToChars"), slug: Some("convertContainedStringsToChars"),
    summary: "Convert string values and strings contained in cells and structs to character vectors.", description: "Recursively convert string values inside cells and structs without changing non-string members.",
    keywords: &["convertContainedStringsToChars", "strings", "core", "matlab", "compatibility"], related: &["string", "char", "convertStringsToChars"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_contained_strings_to_chars.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_contained_strings_to_chars.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "nested typed structure and resident passthrough", location: "builtins::strings::core::compat::tests" }], notes: &["All substantive sidecar signatures, behaviors, example, related links, and source disposition reconciled; no GPU-gather behavior is invented."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
