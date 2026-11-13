use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Unicode digits", paragraphs: &["`digitsPattern` greedily matches one or more Unicode digit characters. `digitsPattern(N)` matches exactly N, and `digitsPattern(minCharacters,maxCharacters)` matches an inclusive range. Positive infinity in the maximum position removes the upper bound.", "Counts accept integer-valued host single/double or any of the eight native integer classes, read exactly from integer storage. Negative, fractional, logical, nonscalar, oversized, reversed-range, and resident controls reject before provider access."] },
    BuiltinDocumentationSection { heading: "Placement", paragraphs: &["Construction produces host pattern metadata in native and browser runtimes. There is no GPU kernel, fusion form, callback, or ambient effect. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-digit-run",
        title: "Find digits in text",
        program: "p = digitsPattern; tf = contains(\"A12\", p)",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true)); assert(isequal(contains(\"ABC\", p), false));",
        },
    },
    BuiltinExample {
        id: "exact-count",
        title: "Specify an exact typed count",
        program: "p = digitsPattern(uint64(2)); tf = contains(\"A12\", p)",
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
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("digitsPattern"), slug: Some("digitsPattern"),
    summary: "Create a pattern matching digit characters.", description: "Create a greedy Unicode digit pattern with optional exact or inclusive bounded counts.",
    keywords: &["digitsPattern", "digits", "strings", "pattern", "matlab", "compatibility"], related: &["string", "char", "pattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/digits_pattern.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/digits_pattern.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "exact classes, ranges and rejected controls", location: "builtins::strings::core::compat::tests" }], notes: &["All three legacy signatures, behaviors, example, links, count classes, GPU restriction, errors, and source field reconciled."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
