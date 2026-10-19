use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Unicode letters", paragraphs: &["`lettersPattern` greedily matches one or more Unicode alphabetic characters by default. One count selects an exact length; two counts select an inclusive range. Positive infinity removes the upper bound.", "Host integer-valued single/double and all eight native integer classes are accepted as exact controls. Negative, fractional, logical, nonscalar, reversed, and oversized controls reject. Automatically resident counts can gather and suspend; explicit `gpuArray` counts reject before provider access."] },
    BuiltinDocumentationSection { heading: "Placement", paragraphs: &["The pattern object is host metadata. No GPU pattern kernel or fusion form exists. In native and browser runtimes, automatic count gathering is the only possible suspension; no callback or ambient service is involved. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-leading-letters",
        title: "Match leading letters",
        program: "p = lettersPattern; tf = startsWith(\"abc1\", p)",
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
        id: "bounded-letters",
        title: "Set an inclusive count range",
        program: "p = lettersPattern(uint8(2), uint16(4)); tf = contains(\"abc1\", p)",
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
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("lettersPattern"), slug: Some("lettersPattern"),
    summary: "Create a pattern matching letter characters.", description: "Create a greedy Unicode-letter pattern with optional exact or inclusive bounded counts.",
    keywords: &["lettersPattern", "letters", "strings", "pattern", "matlab", "compatibility"], related: &["string", "char", "pattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/letters_pattern.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/letters_pattern.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "bounded counts and residency", location: "builtins::strings::core::patterns::tests" }], notes: &["All legacy signatures, count forms, integer exactness, residency split, example, links, and source field reconciled."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
