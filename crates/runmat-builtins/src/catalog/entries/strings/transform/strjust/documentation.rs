use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Alignment", paragraphs: &[
        "strjust(text, side) repositions non-whitespace characters within each text row while preserving row width. The default alignment is right; use left or center to choose the other alignments.",
        "String arrays preserve their shape, two-dimensional character rows preserve their width, and cell arrays preserve their outer shape. N-D character arrays remain a compatibility limitation: the current row-based Runtime producer reconstructs a 2-D character array rather than preserving page geometry. Numeric and integer arrays are not interpreted as character codes and reject before provider access.",
        "The current compatibility implementation treats an unrecognized textual side as right alignment. The first optional side is used; additional trailing arguments are ignored. This behavior is retained rather than presenting those arguments as a separate documented form.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "Justification is host text processing with no provider kernel or fusion route. Numeric and provider-resident inputs reject before provider access.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "left",
    title: "Justify left",
    program: "out = strjust(\"  x\", \"left\")",
    display_output: Some("out = \"x  \""),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out, \"x  \"));",
    },
}];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strjust"), slug: Some("strjust"), summary: "Justify text rows left, right or center.", description: "strjust redistributes leading and trailing whitespace while preserving text-row width and container shape.",
    keywords: &["strjust", "justify", "character", "string", "strings", "transform", "matlab", "compatibility"], related: &["string", "char", "deblank"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text justification", location: "builtins::strings::transform::compat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "typed shape and admission inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strjust/inference.rs" }], notes: &["Finite output ownership is not yet complete for all string, character and nested cell forms."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
