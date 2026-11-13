use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("categorical"), slug: Some("categorical"),
 summary: "Create a categorical array object.", description: "`categorical(A)` maps input labels to integer category codes and stores category labels in object metadata.",
 keywords: &["categorical", "categories", "table", "ordinal"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"String, char, cellstr, numeric, and logical inputs are supported; numeric input includes all eight integer classes.",
"`categorical(A)` derives sorted unique categories from non-missing labels, and numeric default names are rounded to five significant figures; values too close to retain unique names reject.",
"Leading and trailing spaces are removed from string and character-vector input values and from textual `valueset` entries.",
"`categorical(A, valueset)` requires `valueset` to have the same class as `A`, uses the supplied values as the category order, and marks unmatched labels as missing.",
"`categorical(A, valueset, catnames)` maps source values through unique `valueset` entries and stores the corresponding category names from `catnames`; repeated category names deliberately merge distinct source values.",
"Missing source values become undefined unless a corresponding missing `valueset` entry and nonmissing category name explicitly maps them.",
"`\"Ordinal\"` and `\"Protected\"` accept logical or numeric scalar zero or one, including all eight integer classes; ordinal categoricals are always protected. Those option words remain positional labels when not followed by a valid flag.",
"Categorical-to-categorical comparisons require matching category lists.",
"Ordinal categorical arrays support equality, relational comparisons, `min`, and `max` by category order.",
"Use `ordinal(...)` as a compatibility constructor for categoricals whose category order is meaningful.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Categorical arrays are host metadata objects. Current public documentation lists GPU code generation rather than interactive gpuArray input; RunMat can gather resident arguments only when RunMat extensions are enabled.",
"This pure host metadata constructor may suspend while gathering retained arguments through their owning provider; it is not an accelerator kernel, fusion operation, or distributed operation. The default canonical Runtime binding is portable to native and Wasm execution.",
"Finite Managed allocation is currently rejected before gather or label/category scratch allocation. Ordinary unbounded execution retains its original authority and cancellation fence before publication; a quoted metadata/workspace owner remains an explicit L25 obligation, not completed by that refusal.",
"Only the documented string/character/cellstr/numeric/logical label forms are supported by this contract. Rich heterogeneous object conversions are not promised.",
],
},
BuiltinDocumentationSection {
 heading: "Syntax and category mapping",
 paragraphs: &[
"`categorical()` returns an empty categorical array with code shape 0-by-0. Character matrices contribute one label per row; string, numeric, logical, and cell input preserve their logical array shape. Cell labels are visited in column-major order, including N-D pages.",
"`categorical(C)` preserves an existing categorical's codes, complete category list (including unused categories), shape, and ordinal/protected flags. New mappings use one-based double codes; undefined observations carry NaN codes. Text category names form a row string array.",
"`categorical(A,valueset,catnames)` requires one name per distinct mapping value before repeated names merge categories. Invalid option pairs, incompatible value-set classes, ambiguous numeric display names, and malformed metadata reject before returning an object.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "create-categories", title: "Create categories", program: "C = categorical({'red'; 'blue'; 'red'})",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(iscategorical(C));\nassert(isequal(C == 'red', [true; false; true]));" },
}
 ],
 example_exemption: None,
 faqs: &[],
 links: &[
BuiltinDocumentationLink { label: "iscategorical", target: BuiltinDocumentationLinkTarget::Builtin("iscategorical") },
BuiltinDocumentationLink { label: "ordinal", target: BuiltinDocumentationLinkTarget::Builtin("ordinal") },
BuiltinDocumentationLink { label: "isordinal", target: BuiltinDocumentationLinkTarget::Builtin("isordinal") }
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/table/mod.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/mod.rs") }],
 verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "crates/runmat-vm/tests/tables_timetables.rs", location: "crates/runmat-vm/tests/tables_timetables.rs" }
 ],
 notes: &["Source examples have independent assertions or a typed expected-error witness; display transcripts are not fabricated.", "Legacy GPU/fusion metadata is normalized to the typed host/fusion boundary contract. No optional feature is introduced; finite allocation remains an explicit L25 obligation."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
