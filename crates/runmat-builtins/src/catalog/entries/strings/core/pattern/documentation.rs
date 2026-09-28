use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Literal matching", paragraphs: &["`pattern(text)` creates a host pattern object whose text is escaped before matching. Metacharacters in the input are literal, not regular-expression operators. Text may be a string scalar or character vector; numeric and resident values reject without implicit conversion or provider access."] },
    BuiltinDocumentationSection { heading: "Use and placement", paragraphs: &["Pattern-aware string operations including `matches`, `contains`, `startsWith`, `endsWith`, `erase`, and `replace` accept the result. Construction has no GPU kernel or fusion form; the object is host metadata. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "literal-metacharacter",
    title: "Treat punctuation literally",
    program: "p = pattern(\"r.n\"); tf = contains(\"r.n\", p)",
    display_output: Some("tf = logical 1"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(tf, true)); assert(isequal(contains(\"run\", p), false));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("pattern"), slug: Some("pattern"),
    summary: "Create a literal string pattern.", description: "Create a pattern that matches literal text after escaping regular-expression metacharacters.",
    keywords: &["pattern", "literal", "strings", "matlab", "compatibility"], related: &["string", "char", "regexpPattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/pattern.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/pattern.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "literal pattern construction", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy behaviors, example, links, GPU restriction, and source location reconciled; the constructor emits a typed pattern object, not a string."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
