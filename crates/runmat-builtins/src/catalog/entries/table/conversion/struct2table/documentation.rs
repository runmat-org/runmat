use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("struct2table"),
    slug: Some("struct2table"),
    summary: "Convert scalar structs or struct arrays into tables.",
    description: "`struct2table(S)` converts struct fields into table variables. Scalar structs become variable columns, while struct arrays become row-oriented tables.",
    keywords: &["struct2table", "struct", "table", "RowNames"],
    related: &["table2struct", "table"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Field insertion order becomes variable order.", "Compatible field values from a struct array become homogeneous table variables. Same-class integer fields retain their exact class and values; incompatible fields remain cell variables.", "`\"VariableNames\"` and `\"RowNames\"` may be supplied as name-value options.", "`\"AsArray\"` keeps scalar struct field values as array-valued table entries and accepts logical or numeric scalar values exactly equal to zero or one."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Struct and table containers are host values.", "Host-side conversion is a fusion boundary."] }],
    examples: &[BuiltinExample { id: "example-1", title: "Convert struct variables", program: "S = struct('A',[1;2],'B',[3;4]); T = struct2table(S)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T)); assert(isequal(T.A,[1;2]));" } }, BuiltinExample { id: "example-2", title: "Preserve scalar struct arrays as entries", program: "S = struct('A',[1;2]); T = struct2table(S,'AsArray',true)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T)); assert(height(T)==1);" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table2struct", target: BuiltinDocumentationLinkTarget::Builtin("table2struct") }, BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/constructors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
