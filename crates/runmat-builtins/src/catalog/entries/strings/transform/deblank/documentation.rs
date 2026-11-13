use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "deblank removes trailing whitespace from each text element. String arrays keep their shape; character arrays keep their row count but their resulting width can change. Cell arrays preserve their outer shape.",
        "Missing string elements remain missing. Numeric and integer inputs reject without implicit text conversion.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "deblank returns host text and has no provider kernel or fusion route. It rejects resident numeric values before provider access; accepted host text is transformed without callbacks or external resources.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "legacy-1",
    title: "Remove trailing spaces",
    program: "out = deblank(\"abc   \")",
    display_output: Some("out = \"abc\""),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out, \"abc\"));",
    },
}];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("deblank"), slug: Some("deblank"),
    summary: "Remove trailing whitespace from text.",
    description: "s = deblank(text) removes trailing whitespace from text while preserving the input container kind.",
    keywords: &["deblank", "strings", "transform", "matlab", "compatibility"], related: &["string", "char"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text and integer admission", location: "builtins::strings::transform::compat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/deblank/inference.rs" }], notes: &["Resident numeric input is rejected before gathering. The host operation and input boundary are covered by the linked runtime tests."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
