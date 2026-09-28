use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Regular-expression matching", paragraphs: &["`regexpPattern(expr)` stores a regular-expression source in a host pattern object without escaping it. Consumers compile and apply the expression when matching. A string scalar or character vector is accepted; numeric and resident values reject without text conversion or provider access."] },
    BuiltinDocumentationSection { heading: "Use and placement", paragraphs: &["Pattern-aware `matches`, `contains`, `startsWith`, `endsWith`, `erase`, `replace`, and boundary helpers accept the result. Construction has no GPU kernel or fusion form. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "digit-regex",
    title: "Match a regular expression",
    program: "p = regexpPattern(\"A\\d+\"); tf = contains(\"A12\", p)",
    display_output: Some("tf = logical 1"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(tf, true)); assert(isequal(contains(\"AB\", p), false));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("regexpPattern"), slug: Some("regexpPattern"),
    summary: "Create a regular expression string pattern.", description: "Create a pattern object backed by regular-expression text.",
    keywords: &["regexpPattern", "regex", "strings", "pattern", "matlab", "compatibility"], related: &["string", "char", "pattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/regexp_pattern.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/regexp_pattern.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "regex-backed pattern construction", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy behavior, example, links, GPU restriction, and source field reconciled. Expression validation belongs to the consuming pattern operation."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
