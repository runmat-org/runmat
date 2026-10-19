use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table2cell"),
    slug: Some("table2cell"),
    summary: "Convert a table into a cell array.",
    description: "`table2cell(T)` returns an `height(T)` by `width(T)` cell array of table element values.",
    keywords: &["table2cell", "table", "cell"],
    related: &["cell2table"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Each table variable contributes its contents to the corresponding output cells.", "Integer elements retain their exact class and value, including wide int64 and uint64 values.", "Multi-column variables are preserved as same-row slices in the corresponding cell."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Conversion runs on host table metadata.", "Host-side conversion is a fusion boundary."] }],
    examples: &[BuiltinExample { id: "example-1", title: "Converting Table Elements To Cells", program: "T = array2table([1 3;2 4]); C = table2cell(T)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(C)); assert(isequal(size(C),[2 2])); assert(C{1,1}==1); assert(C{2,2}==4);" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "cell2table", target: BuiltinDocumentationLinkTarget::Builtin("cell2table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/extractors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
