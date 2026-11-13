use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("dictionary"), slug: Some("dictionary"),
 summary: "Create a key-value dictionary object.", description: "`dictionary(keys, values)` creates a host dictionary object. Alternating key/value pairs are also accepted, including array-valued pairs.",
 keywords: &["dictionary", "map", "keys", "values", "containers.Map"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Keys and values must have the same size unless the value input is scalar, in which case it expands to every key.",
"String, numeric, logical, char, and cell collections are expanded into entries and normalized to one compatible configured key class and one compatible configured value class.",
"All eight integer classes remain exact, including `uint64` keys beyond binary64's exact range. Construction, lookup, assignment, and removal compare authoritative configured-class integers.",
"Duplicate keys keep their first insertion position and the last supplied value.",
"Multi-key lookup preserves the configured uniform value class and the query shape; scalar lookup returns a scalar. Non-scalar assignment values must match the key-query shape, while scalar assignment expands.",
"Resident keys or values are a RunMat-only interactive extension. Compatibility mode rejects them before provider access; RunMat mode gathers them exactly through the owning provider.",
"Concrete MATLAB dictionary class metadata and unsupported heterogeneous object-type conversions remain recorded general gaps.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Dictionary objects are host metadata containers. Resident input is a RunMat-only extension gated before exact download through the owning provider.",
"Construction is a pure, potentially suspending host operation. The canonical default Runtime binding gathers retained values through their owning provider after extension admission, is a fusion boundary, and links portably on native and Wasm.",
"Finite Managed metadata and entry-list scratch are not yet owned by exact quotes; such invocations reject before gather or raw allocation. Ordinary unbounded execution checks its original allocation/cancellation authority before publication. This is a recorded L25 limit, not finite ownership completion.",
],
},
BuiltinDocumentationSection {
 heading: "Construction, shape, and mutation",
 paragraphs: &[
"`dictionary()` creates an empty host dictionary. A two-argument keys/values form expands each supported collection into entries; additional alternating arguments each contribute a key/value pair. Odd argument counts reject.",
"Cell entries are enumerated in logical column-major order. Multi-key lookups retain the entire query shape, including N-D pages; fallback cell results are built using the canonical cell column-major boundary.",
"The configured real numeric class is authoritative for comparison and publication. Single entries retain native binary32 payloads, scalar lookup retains their class, and homogeneous multi-key lookup returns a Single array with the query shape. Conversion of integer queries and assigned values to Single uses native integer-to-binary32 rounding. Complex arrays and rich object payloads are not promised as per-element entry expansion. Mixed unsupported class conversions reject rather than establishing a second class authority.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "create-dictionary", title: "Create a dictionary", program: "D = dictionary([\"a\" \"b\"], [1 2])",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(D(\"a\") == 1);\nassert(D(\"b\") == 2);" },
}
 ],
 example_exemption: None,
 faqs: &[],
 links: &[
BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Builtin("containers.Map") }
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/table/containers/dictionary.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/containers/dictionary.rs") }],
 verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "crates/runmat-vm/tests/tables_timetables.rs", location: "crates/runmat-vm/tests/tables_timetables.rs" }
 ],
 notes: &["Source examples have independent assertions or a typed expected-error witness; display transcripts are not fabricated.", "Legacy GPU/fusion metadata is normalized to the typed host/fusion boundary contract. No optional feature is introduced; finite allocation remains an explicit L25 obligation."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
