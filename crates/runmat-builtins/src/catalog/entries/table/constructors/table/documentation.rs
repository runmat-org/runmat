use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("table"), slug: Some("table"),
 summary: "Create a table from column variables or preallocate numeric table variables.", description: "`table(var1, ..., varN)` stores arrays with the same row count as named table variables. `table(Size=sz, VariableTypes=types)` preallocates numeric variables from a two-element size vector.",
 keywords: &["table", "VariableNames", "RowNames", "Size", "VariableTypes"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Each input variable keeps its own class, shape after the row dimension, and exact values.",
"Preallocation currently supports `double`, `single`, and all eight built-in integer variable types. Integer columns are initialized with same-class zeros.",
"`Size` accepts floating or any built-in integer class, must contain exactly two nonnegative integer values, and must request the same number of variables listed by `VariableTypes`.",
"`VariableNames` and `RowNames` can label the variables and rows. Names must satisfy the table naming and uniqueness rules.",
"Other documented preallocation types, including logical, string, categorical, datetime, duration, and calendar duration, remain implementation gaps.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Tables are host containers. Automatically resident variables may gather transparently; explicit GPU variables require RunMat extension mode.",
"The canonical default binding roots retained arguments across provider gathers and preserves source class/payload owners. This pure host constructor may suspend, is a fusion boundary, and uses portable Runtime execution on native and Wasm; no table GPU kernel is advertised.",
"Finite Managed construction is rejected before raw preallocation, name/schema scratch, or gather. A complete quoted column/metadata allocation vertical remains L25 work. Other documented preallocation classes and richer empty-table metadata are explicit unsupported implementation frontiers, not inferred as implemented.",
],
},
BuiltinDocumentationSection {
 heading: "Constructor and schema",
 paragraphs: &[
"`table()` creates an empty host table. Data-variable construction accepts heterogeneous supported arrays but requires an equal leading row count; trailing dimensions and native payloads are retained. `VariableNames` defaults to Var1, Var2, and so on.",
"`Size` and `VariableTypes` must occur together and cannot be mixed with data variables. Variable type and explicit variable-name counts must match the requested width. Size must be a real two-element nonnegative integral vector in platform range; unsupported variable types reject.",
"Variable and row labels follow existing table naming rules, including uniqueness and reserved-name validation. Empty and zero-width tables retain existing table metadata limitations; positive height without any variables is not represented by a separate table-height owner.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "existing-variables", title: "Creating A Table From Existing Variables", program: "age = int16([31; 42]);\nscore = single([0.8; 0.9]);\nT = table(age, score, VariableNames=[\"Age\" \"Score\"])",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T));\nassert(isa(T.Age,'int16'));\nassert(isa(T.Score,'single'));\nassert(isequal(T.Age,int16([31;42])));\nassert(height(T)==2 && width(T)==2);" },
},
BuiltinExample {
 id: "preallocate-integers", title: "Preallocating Integer Variables", program: "T = table(Size=uint16([3 2]), VariableTypes=[\"int32\" \"uint64\"], VariableNames=[\"Count\" \"Identifier\"])",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T));\nassert(isa(T.Count,'int32'));\nassert(isa(T.Identifier,'uint64'));\nassert(isequal(T.Count,int32(zeros(3,1))));\nassert(isequal(T.Identifier,uint64(zeros(3,1))));" },
}
 ],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Do integer variables lose precision when stored in a table?", answer: "No. Each integer variable retains its exact signedness, width, shape, and values, including int64 and uint64 values above `flintmax`." },
BuiltinDocumentationFaq { question: "Can a table contain variables of different types?", answer: "Yes. Variables can use different supported classes and widths as long as they have the same number of rows." }],
 links: &[
BuiltinDocumentationLink { label: "array2table", target: BuiltinDocumentationLinkTarget::Builtin("array2table") },
BuiltinDocumentationLink { label: "table2array", target: BuiltinDocumentationLinkTarget::Builtin("table2array") },
BuiltinDocumentationLink { label: "table2cell", target: BuiltinDocumentationLinkTarget::Builtin("table2cell") },
BuiltinDocumentationLink { label: "table2struct", target: BuiltinDocumentationLinkTarget::Builtin("table2struct") }
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/table/builtins/constructors.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors.rs") }],
 verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integer_table_and_tangent_semantics", location: "crates/runmat-vm/tests/tables_timetables.rs" }
 ],
 notes: &["Source examples have independent assertions or a typed expected-error witness; display transcripts are not fabricated.", "Legacy GPU/fusion metadata is normalized to the typed host/fusion boundary contract. No optional feature is introduced; finite allocation remains an explicit L25 obligation."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
