use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("array2table"),
    slug: Some("array2table"),
    summary: "Convert an array into a table.",
    description: "`array2table(A)` splits a homogeneous two-dimensional array into table variables by column. Native integer columns preserve the exact class and values of `A`.",
    keywords: &["array2table", "table", "VariableNames", "RowNames", "DimensionNames"],
    related: &["table2array", "table"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Numeric, complex, logical, string, and cell arrays are split column-wise.", "`\"VariableNames\"`, `\"RowNames\"`, and `\"DimensionNames\"` are supported. Current variable names may contain Unicode, spaces, and leading or trailing whitespace; row-name edge whitespace is removed.", "Interactive resident GPU input is a mode-gated RunMat extension. It gathers exact typed storage before table construction and rejects in MATLAB-compatible mode."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Inputs are gathered before table materialization.", "Host-side conversion is a fusion boundary."] }],
    examples: &[BuiltinExample { id: "example-1", title: "Create named variables", program: "T = array2table([1 3; 2 4], 'VariableNames', {'A','B'})", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T)); assert(height(T)==2); assert(width(T)==2); assert(isequal(T.A,[1;2])); assert(isequal(T.B,[3;4]));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table2array", target: BuiltinDocumentationLinkTarget::Builtin("table2array") }, BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/constructors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
