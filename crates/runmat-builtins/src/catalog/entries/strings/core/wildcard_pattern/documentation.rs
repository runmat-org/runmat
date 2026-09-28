use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Lazy wildcard", paragraphs: &["`wildcardPattern` matches arbitrary text, including zero characters, and is lazy toward the shortest match. An exact count or inclusive minimum/maximum counts can be provided. Positive infinity removes the upper bound.", "Host integer-valued single/double and all eight native integer classes are accepted as exact controls. Negative, fractional, logical, nonscalar, reversed, and oversized controls reject. Automatically resident counts can gather and suspend; explicit `gpuArray` counts reject before provider access."] },
    BuiltinDocumentationSection { heading: "Placement", paragraphs: &["The pattern object is host metadata, with no GPU pattern kernel or fusion form. Automatic count gathering is the only possible suspension; no callback or ambient service is involved. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-arbitrary-text",
        title: "Match arbitrary text",
        program: "p = wildcardPattern; tf = contains(\"abc\", p)",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "exact-wildcard",
        title: "Match an exact number of characters",
        program: "p = wildcardPattern(uint16(2)); tf = contains(\"abc\", p)",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("wildcardPattern"), slug: Some("wildcardPattern"),
    summary: "Create a pattern matching arbitrary text.", description: "Create a lazy wildcard pattern with optional exact or inclusive bounded counts.",
    keywords: &["wildcardPattern", "wildcard", "strings", "pattern", "matlab", "compatibility"], related: &["string", "char", "pattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/wildcard_pattern.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/wildcard_pattern.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "lazy bounds and resident count routing", location: "builtins::strings::core::patterns::tests" }], notes: &["All legacy signatures, lazy behavior, count classes, residency split, example, links, and source field reconciled."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
