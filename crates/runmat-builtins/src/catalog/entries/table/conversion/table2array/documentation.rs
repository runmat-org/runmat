use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table2array"),
    slug: Some("table2array"),
    summary: "Convert table variables into a homogeneous array when possible.",
    description: "`table2array(T)` horizontally concatenates compatible table variables into one homogeneous array. Table row names, timetable row times, and other table properties are not included.",
    keywords: &["table2array", "table", "array"],
    related: &["array2table"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Same-class integer variables concatenate without conversion, preserving exact signedness, width, and values above `flintmax`.", "Compatible mixed numeric variables use the table-concatenation dominant class and its assignment conversion rules. A native single variable makes an otherwise floating result single.", "If all variables are cell arrays, the result is a cell array equivalent to `table2cell(T)`. Broader heterogeneous concatenation remains limited by the currently implemented table indexing surface."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["The table container is host-side. Nested resident variables are not gathered solely by this conversion; unsupported heterogeneous combinations can remain cell-valued with resident elements.", "Host-side conversion is a fusion boundary."] }],
    examples: &[BuiltinExample { id: "example-1", title: "Converting Numeric Variables To A Matrix", program: "A = table2array(array2table([1 3; 2 4]))", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(A,[1 3;2 4]));" } }, BuiltinExample { id: "example-2", title: "Preserving Wide Integer Values", program: "T = table(uint64([0x0020000000000001u64;0x0020000000000003u64])); A = table2array(T)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A,'uint64')); assert(isequal(A,uint64([0x0020000000000001u64;0x0020000000000003u64])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "array2table", target: BuiltinDocumentationLinkTarget::Builtin("array2table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/extractors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
