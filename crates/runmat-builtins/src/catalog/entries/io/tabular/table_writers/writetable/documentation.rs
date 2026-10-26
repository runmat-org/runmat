use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("writetable"), slug: Some("writetable"),
 summary: "Write a table to a delimited text file.", description: "`writetable(T, filename)` writes table variables to a delimited text file.",
 keywords: &["writetable", "table", "csv", "delimited text", "WriteVariableNames"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"`\"Delimiter\"` controls the field separator and defaults to comma.",
"`\"WriteVariableNames\"` controls header emission and defaults to true.",
"Fields containing delimiters, quotes, or newlines are CSV-quoted.",
"One-column numeric variables and scalar numeric cell values use general formatting with 15 significant digits for double storage and 7 for single storage. Scalar logical fields use 1 or 0; typed integers retain exact decimal values.",
"Variables from all eight native integer classes are serialized with their exact decimal values.",
"A typed-integer `WriteVariableNames` value, explicit `gpuArray` input, and the bytes-written output are available in RunMat mode.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Table data is serialized on the host. Automatically resident values gather transparently; explicit gpuArray input requires RunMat mode.",
"This effectful host operation reads the selected environment and writes the filesystem; it may suspend and throw, requires Filesystem capability, and is a fusion boundary. Native and browser/Wasm execution require a selected provider implementing the existing staged publication contract; there is no ambient filesystem fallback from an explicit Runtime context.",
"Finite Managed execution is refused before gather, serialization buffers, or filesystem effects because complete quoted text/row/metadata scratch ownership is not yet implemented. That explicit L25 limit is not allocation completion. Spreadsheet/formatting APIs, append/write modes, row-name emission, and broader table I/O options are not implemented by this delimited writer.",
],
},
BuiltinDocumentationSection {
 heading: "Syntax and selected publication",
 paragraphs: &[
"`writetable(T,filename)` accepts an exact table object, not a timetable or arbitrary object. Name-value pairs are case-insensitive. Delimiter accepts scalar text; WriteVariableNames accepts the existing logical/numeric/text Boolean controls. Unknown or incomplete option pairs reject.",
"The filename must be nonempty scalar text. The canonical selected filesystem provider and working directory are captured before operand gather, and retained with their resource lease until staged replacement completes. Table variables are borrowed from the rooted table rather than copied into an alternate column list.",
"Output arity and extension admission precede effects. Class, options, field layout, byte-count exactness, and serialization are checked before publication. A RunMat bytes-written result is an exact double scalar; the MATLAB-compatible form requests no output.",
"Argument errors use RunMat:writetable:InvalidArgument. Filesystem/publication failures use RunMat:writetable:IOError and retain their source; cancellation and authority/control errors retain their original identity. These terminal failures never trigger GPU gather retry.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "write-csv", title: "Write CSV", program: "T = table(int16([3;4]), 'VariableNames', {'Count'});\nwritetable(T, 'out.csv');\ncontents = fileread('out.csv');",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::NativeFilesystem,
 fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
 id: BuiltinExampleFixtureId { local_name: "write-csv" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
 entries: &[BuiltinFilesystemEntry::File { relative_path: "out.csv", content: BuiltinFixtureContent::Utf8("old-data\n") }],
 }),
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(contents, sprintf('Count\\n3\\n4\\n')));" },
}
 ],
 example_exemption: None,
 faqs: &[],
 links: &[
BuiltinDocumentationLink { label: "MATLAB language extensions", target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions") },
BuiltinDocumentationLink { label: "readtable", target: BuiltinDocumentationLinkTarget::Builtin("readtable") },
BuiltinDocumentationLink { label: "writetimetable", target: BuiltinDocumentationLinkTarget::Builtin("writetimetable") }
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
