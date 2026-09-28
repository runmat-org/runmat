use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Boundary types", paragraphs: &["`textBoundary` and `textBoundary(\"either\")` match the start or end of a text scalar. `\"start\"` selects only the beginning and `\"end\"` only the end. A string scalar or character vector is accepted as the type; names are case-insensitive.", "The result is a regex-backed host pattern used by `contains`, `startsWith`, `endsWith`, `erase`, and `replace`. Full pattern-expression algebra, including addition and negation, is outside this constructor's current behavior."] },
    BuiltinDocumentationSection { heading: "Placement", paragraphs: &["Construction uses host pattern metadata, not a GPU kernel or fusion. Evaluating a supplied type may gather automatically resident data and suspend; pattern-aware consumers separately handle their text inputs. At most one output is available."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-start",
        title: "Create a start boundary",
        program: "p = textBoundary(\"start\"); out = replace(\"abc\", p, \">\")",
        display_output: Some("out = \">abc\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out, \">abc\"));",
        },
    },
    BuiltinExample {
        id: "legacy-end",
        title: "Create an end boundary",
        program: "p = textBoundary(\"end\"); out = replace(\"abc\", p, \"<\")",
        display_output: Some("out = \"abc<\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out, \"abc<\"));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("textBoundary"), slug: Some("textBoundary"),
    summary: "Create a pattern matching the start or end of text.", description: "Create a host pattern for either, start-only, or end-only text boundaries.",
    keywords: &["textBoundary", "strings", "pattern", "boundary", "matlab", "compatibility"], related: &["pattern", "regexpPattern", "lettersPattern"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does textBoundary support pattern negation?", answer: "No. This constructor returns a boundary pattern object; full pattern-expression algebra remains separate work." },
        BuiltinDocumentationFaq { question: "Does textBoundary execute on the GPU?", answer: "No. It constructs host pattern metadata. Pattern-aware consumers handle any resident text input separately." },
    ],
    links: &[BuiltinDocumentationLink { label: "pattern", target: BuiltinDocumentationLinkTarget::Builtin("pattern") }, BuiltinDocumentationLink { label: "regexpPattern", target: BuiltinDocumentationLinkTarget::Builtin("regexpPattern") }, BuiltinDocumentationLink { label: "lettersPattern", target: BuiltinDocumentationLinkTarget::Builtin("lettersPattern") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/text_boundary.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/text_boundary.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "boundary selectors and replace consumer", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy custom examples, FAQs, three related links, limitations, GPU note, and source field reconciled. The missing legacy typed signature is now explicit."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
