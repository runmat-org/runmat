use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("upsample"),
 slug: Some("upsample"),
 summary: "Increase a signal's sample rate by inserting zeros.",
 description: "`y = upsample(x,n)` inserts `n-1` zeros between samples. `y = upsample(x,n,phase)` offsets the original samples by zero-based `phase`.",
 keywords: &["upsample",
 "sample rate",
 "zero insertion",
 "signal processing"],
 related: &["downsample",
 "resample",
 "filter",
 "stem"],

 sections: &[BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &["Vectors and matrices are documented; N-D inputs are a compatibility-gated RunMat extension. The first nonsingleton dimension is expanded.",
 "`N` is a positive integer-valued single/double scalar. `PHASE` defaults to zero and satisfies `0 <= PHASE < N`; typed controls are independently gated exact RunMat extensions.",
 "Integer samples preserve all eight classes exactly and inserted zeros use the same native storage class."] }
,
 BuiltinDocumentationSection {
 heading: "GPU execution",
 paragraphs: &["A proven owner may allocate zeroed storage and scatter exact input samples. Owner, device, shape, storage and class metadata are validated before publication.",
 "Unknown-device inputs reject before provider work; admitted provider errors are terminal unless certified unsupported before effects."] }
],

 examples: &[BuiltinExample {
 id: "example-1",
 title: "Insert zeros",
 program: "x = [1 2 3];\ny = upsample(x,2)",
 display_output: Some("y = [1 0 2 0 3 0]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(y,[1 0 2 0 3 0]));" }
 }
,
 BuiltinExample {
 id: "example-2",
 title: "Offset samples",
 program: "x = [1 2 3];\ny = upsample(x,3,1)",
 display_output: Some("y = [0 1 0 0 2 0 0 3 0]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(y,[0 1 0 0 2 0 0 3 0]));" }
 }
],
 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq {
 question: "Which dimension is expanded?",
 answer: "The first nonsingleton dimension." }
,
 BuiltinDocumentationFaq {
 question: "Does upsample interpolate?",
 answer: "No. It only inserts zeros." }
,
 BuiltinDocumentationFaq {
 question: "Can GPU tensors stay resident?",
 answer: "Yes when their proven owner supports exact zero-allocation and scatter." }
],

 links: &[BuiltinDocumentationLink {
 label: "downsample",
 target: BuiltinDocumentationLinkTarget::Builtin("downsample") }
,
 BuiltinDocumentationLink {
 label: "resample",
 target: BuiltinDocumentationLinkTarget::Builtin("resample") }
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
