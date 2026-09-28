use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cell2table"),
    slug: Some("cell2table"),
    summary: "Convert a cell array into a table.",
    description: "`cell2table(C)` converts each cell-array column into a table variable.",
    keywords: &["cell2table", "cell", "table", "VariableNames"],
    related: &["table2cell"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`\"VariableNames\"` and `\"RowNames\"` are supported.", "Compatible integer one-row values in each column are concatenated into a homogeneous variable; incompatible values remain a column cell array.", "Same-class integer columns preserve exact native storage for all eight classes. Mixed integer columns use the leftmost integer class with saturating conversion; scalar doubles may coexist except with `int64` or `uint64`."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Cell and table containers are host values. Nested resident objects remain intact; resident numeric rows are currently left as cell-valued variables rather than concatenated.", "Host-side conversion is a fusion boundary."] }],
    examples: &[BuiltinExample { id: "example-1", title: "Convert cells", program: "T = cell2table({1, 'a'; 2, 'b'}, 'VariableNames', {'A','B'})", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T)); assert(height(T)==2); assert(width(T)==2); assert(iscell(T.A)); assert(T.A{1}==1);" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table2cell", target: BuiltinDocumentationLinkTarget::Builtin("table2cell") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/constructors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
