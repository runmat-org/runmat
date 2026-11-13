use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Inputs are string arrays, character vectors, or cell arrays of character vectors. A string input selects string output; otherwise a cell input selects cellstr output; all-character-vector input returns a character vector.",
        "String and cell arrays combine by compatible-size implicit expansion. Character vectors act as scalar text and can combine with every compatible array size.",
        "Trailing whitespace is preserved in every supported text representation. Numeric, logical and integer values are not implicitly converted to text; all eight integer classes reject.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "append is a portable, pure host text operation with no provider hook or GPU fusion. Resident numeric handles reject before provider access or gather. Runtime errors may occur for invalid inputs or incompatible sizes.",
        "The default runtime binding is shared by interpreted and compiled calls. There are no callbacks or external resources.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-1", title: "Append string scalars", program: "out = append(\"run\", \"mat\")", display_output: Some("out = \"runmat\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(out, \"runmat\"));" } },
    BuiltinExample { id: "legacy-2", title: "Preserve trailing spaces in character vectors", program: "out = append('Hello ', 'World')", display_output: Some("out = 'Hello World'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(out)); assert(strcmp(out, 'Hello World'));" } },
    BuiltinExample { id: "legacy-3", title: "Expand a string column against a row", program: "out = append([\"A\"; \"B\"], [\"1\" \"2\"])", display_output: Some("out = [\"A1\" \"A2\"; \"B1\" \"B2\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [2 2])); assert(isequal(out, [\"A1\" \"A2\"; \"B1\" \"B2\"]));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("append"), slug: Some("append"),
    summary: "Append text inputs elementwise.",
    description: "s = append(arg1, ...) combines string arrays, character vectors, or cell arrays of character vectors elementwise. A string input selects string output; otherwise a cell input selects cellstr output; otherwise all-character-vector input produces a character vector.",
    keywords: &["append", "strings", "transform", "matlab", "compatibility"],
    related: &["string", "char"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "append text and integer admission", location: "builtins::strings::transform::compat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/append/inference.rs" }], notes: &["Example assertions check the resulting text and shape. Typed signatures describe accepted inputs, while the placement contract records host execution."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
