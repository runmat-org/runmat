use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("downsample"),
 slug: Some("downsample"),
 summary: "Decrease a signal's sample rate by keeping every Nth sample.",
 description: "`y = downsample(x,n)` keeps every `n`th sample starting at the first sample. `y = downsample(x,n,phase)` starts at zero-based `phase`.",

 keywords: &["downsample",
 "sample rate",
 "decimation",
 "signal processing"],
 related: &["upsample",
 "resample",
 "filter",
 "stem"],

 sections: &[
  BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &["Vectors and matrices are documented; N-D inputs are a compatibility-gated RunMat extension. The first nonsingleton dimension is selected.",
 "`N` is a positive integer-valued single/double scalar. `PHASE` defaults to zero and satisfies `0 <= PHASE < N`; typed controls are independently gated exact RunMat extensions.",
 "Integer samples preserve all eight classes exactly. No anti-aliasing filter is applied."] }
,

  BuiltinDocumentationSection {
 heading: "GPU execution",
 paragraphs: &["A proven owner may execute storage-aware linear gather. The returned owner, device, shape, storage, integer/logical class and floating precision are validated.",
 "Unknown-device inputs reject before provider work. Before a provider hook, an unavailable route may materialize through its proven owner. A legacy hook error with unknown effect timing is terminal; malformed outputs are retired only through their proven owner."] }
,

 ],

 examples: &[
  BuiltinExample {
 id: "example-1",
 title: "Keep every second sample",
 program: "x = [1 2 3 4 5];\ny = downsample(x,2)",
 display_output: Some("y = [1 3 5]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(y,[1 3 5])); assert(isequal(size(y),[1 3]));" }
 }
,

  BuiltinExample {
 id: "example-2",
 title: "Use phase one",
 program: "x = [1 2 3 4 5];\ny = downsample(x,2,1)",
 display_output: Some("y = [2 4]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(y,[2 4]));" }
 }
,

  BuiltinExample {
 id: "example-3",
 title: "Downsample matrix columns",
 program: "X = [1 3;2 4;5 7];\nY = downsample(X,2)",
 display_output: Some("Y = [1 3; 5 7]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(Y,[1 3;5 7]));" }
 }
,

 ],
 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq {
 question: "Which dimension is used?",
 answer: "The first nonsingleton dimension." }
,
 BuiltinDocumentationFaq {
 question: "Is anti-aliasing applied?",
 answer: "No. Apply a low-pass filter separately." }
,
 BuiltinDocumentationFaq {
 question: "Can GPU tensors be used?",
 answer: "Yes, through a proven owning provider with exact output validation. Unphased hook failures are terminal rather than replayed." }
],

 links: &[BuiltinDocumentationLink {
 label: "upsample",
 target: BuiltinDocumentationLinkTarget::Builtin("upsample") }
,
 BuiltinDocumentationLink {
 label: "filter",
 target: BuiltinDocumentationLinkTarget::Builtin("filter") }
,
 BuiltinDocumentationLink {
 label: "stem",
 target: BuiltinDocumentationLinkTarget::Builtin("stem") }
,
 BuiltinDocumentationLink {
 label: "Source",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate.rs") }
],
 media: &[],

 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink {
 label: "Sample-rate execution and provider validation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate.rs") }
,
 BuiltinDocumentationLink {
 label: "Controls",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate/controls.rs") }
,
 BuiltinDocumentationLink {
 label: "Numerical kernels",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate/numerics.rs") }
,
 BuiltinDocumentationLink {
 label: "Provider adaptation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate/provider.rs") }
],
 verification: &[BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Host, integer, extension and provider semantics",
 location: "builtins::math::signal::sample_rate::tests" }
],
 notes: &[] }
,
 introduced: None,
 status: None,

}
;
