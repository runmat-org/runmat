use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table2struct"),
    slug: Some("table2struct"),
    summary: "Convert a table into row structs or a scalar struct of variables.",
    description: "`table2struct(T)` converts each table row into a struct with fields matching table variable names. `table2struct(T, \"ToScalar\", true)` returns one scalar struct whose fields contain whole table variables.",
    keywords: &["table2struct", "table", "struct", "ToScalar"],
    related: &["struct2table"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Variable order is preserved.", "The default conversion returns one row struct per table row; integer fields retain their exact scalar class and value.", "`\"ToScalar\"` keeps every complete table variable in one scalar struct without converting integer storage.", "The documented logical `ToScalar` selector works in MATLAB-compatible mode. Typed-integer selector values are available only when RunMat extensions are enabled."] }, BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Table and struct containers are host values.", "Host-side conversion is a fusion boundary."] }],
    examples: &[
        BuiltinExample { id: "example-1", title: "Convert rows to structs", program: "T = array2table([1 3;2 4],'VariableNames',{'A','B'}); S = table2struct(T)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(S)==2); assert(S(1).A==1); assert(S(2).B==4);" } },
        BuiltinExample { id: "example-2", title: "Convert variables to one scalar struct", program: "T = array2table([1 3;2 4],'VariableNames',{'A','B'}); S = table2struct(T,'ToScalar',true)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(S)); assert(isequal(S.A,[1;2])); assert(isequal(S.B,[3;4]));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "struct2table", target: BuiltinDocumentationLinkTarget::Builtin("struct2table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/conversions/extractors.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table conversion tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM table integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }], notes: &["Legacy source fields are preserved in catalog documentation; display-only examples now have assertions.", "Legacy gpu_support and fusion map to typed placement; requires_feature was null and tested maps to verification evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
