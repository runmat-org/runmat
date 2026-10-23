use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("writetimetable"), slug: Some("writetimetable"),
 summary: "Write a timetable to a delimited text file.", description: "`writetimetable(TT, filename)` writes timetable row times and variables to a delimited text file.",
 keywords: &["writetimetable", "timetable", "csv", "RowTimes"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Row times are emitted as a leading `Time` field.",
"`\"Delimiter\"` and `\"WriteVariableNames\"` are supported.",
"Variables from all eight native integer classes are serialized with their exact decimal values.",
"A typed-integer `WriteVariableNames` value, explicit `gpuArray` input, and the bytes-written output are available in RunMat mode.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Timetable data is serialized on the host. Automatically resident values gather transparently; explicit gpuArray input requires RunMat mode.",
"The canonical default binding reads selected environment state, writes the filesystem, may suspend/throw, and requires Filesystem capability. It gathers host serialization inputs only after extension admission and remains a fusion boundary. Native and Wasm/browser paths share the selected provider/staged publication contract, not a parallel filesystem authority.",
"Finite Managed serialization rejects before gather or effects because quoted row/text/metadata workspace ownership remains an explicit L25 gap. The guard is not completion. This entry does not promise spreadsheet output, append/write modes, calendar-format controls, or other unimplemented tabular writer options.",
],
},
BuiltinDocumentationSection {
 heading: "Syntax and selected publication",
 paragraphs: &[
"`writetimetable(TT,filename)` requires an exact timetable object with RowTimes metadata; ordinary tables reject. Row times are read from the rooted timetable and emitted before data variables under the Time header. Existing row-value formatting is retained; this does not add a calendar-format option.",
"Delimiter accepts scalar text and WriteVariableNames controls both the Time header and variable headers. Unknown or incomplete name-value pairs, inconsistent variable/row-time heights, and malformed storage reject before publication.",
"The exact selected filesystem provider and working directory are captured before gather and retained through the shared staged replacement owner. Arity and compatibility gates precede effects; a RunMat-only bytes-written result must be an exactly representable double.",
"Argument errors use RunMat:writetimetable:InvalidArgument, filesystem/publication errors use RunMat:writetimetable:IOError with the original source, and language/control failures retain their original identities. No terminal error authorizes a GPU gather retry.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "write-timetable", title: "Write timetable", program: "TT = timetable(seconds([0;1]), int16([3;4]), 'VariableNames', {'Count'});\nwritetimetable(TT, 'out.csv');\ncontents = fileread('out.csv');",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::NativeFilesystem,
 fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
 id: BuiltinExampleFixtureId { local_name: "write-timetable" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
 entries: &[BuiltinFilesystemEntry::File { relative_path: "out.csv", content: BuiltinFixtureContent::Utf8("old-data\n") }],
 }),
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(contains(contents,'Time,Count'));\nassert(contains(contents,',3'));\nassert(contains(contents,',4'));\nassert(~contains(contents,'old-data'));" },
}
 ],
 example_exemption: None,
 faqs: &[],
 links: &[
BuiltinDocumentationLink { label: "MATLAB language extensions", target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions") },
BuiltinDocumentationLink { label: "writetable", target: BuiltinDocumentationLinkTarget::Builtin("writetable") },
BuiltinDocumentationLink { label: "readtimetable", target: BuiltinDocumentationLinkTarget::Builtin("readtimetable") }
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/table/mod.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/mod.rs") }],
 verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }
 ],
 notes: &["Source examples have independent assertions or a typed expected-error witness; display transcripts are not fabricated.", "Legacy GPU/fusion metadata is normalized to the typed host/fusion boundary contract. No optional feature is introduced; finite allocation remains an explicit L25 obligation."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
