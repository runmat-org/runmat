use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`newline` returns a string scalar containing one line-feed character (`char(10)`). It takes no inputs."] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &["The value is created on the host in native and browser runtimes; no GPU input, kernel, fusion form, callback, workspace, filesystem, network, randomness, or time effect is involved. At most one output is supported."] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "legacy-example",
    title: "Get a line feed",
    program: "s = newline",
    display_output: Some("s is a string scalar containing a line feed"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isStringScalar(s)); assert(isequal(char(s), char(10)));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("newline"), slug: Some("newline"),
    summary: "Return a newline string scalar.", description: "`newline` returns a scalar string containing one line-feed character.",
    keywords: &["newline", "strings", "core", "matlab", "compatibility"], related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/newline.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/newline.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "newline construction", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy title, summary, signature, example, two related links and source field reconciled; the legacy GPU-gather note was inapplicable because newline takes no input."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
