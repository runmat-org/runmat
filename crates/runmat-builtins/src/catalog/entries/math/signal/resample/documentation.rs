use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("resample"),
 slug: Some("resample"),
 summary: "Resample a signal by a rational factor using FIR antialiasing.",
 description: "`y = resample(x,p,q)` changes the sample rate by `p/q`. Optional forms supply the filter design half-length and Kaiser beta, a coefficient vector, `Dimension`, and a second output containing the used filter.",
 keywords: &["resample",
 "sample rate",
 "polyphase",
 "antialiasing",
 "FIR",
 "signal processing"],
 related: &["upsample",
 "downsample",
 "filter",
 "fir1"],

 sections: &[BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &["Input X is dense single or double, real or complex; logical samples are admitted and converted to double. P and Q are positive integer scalars and are reduced by their greatest common divisor.",
 "The default antialiasing filter uses half-length 10 and Kaiser beta 5. A supplied coefficient vector is used directly. Dimension selects the operating axis; otherwise the first nonsingleton dimension is used.",
 "The result preserves single versus double and real versus complex class. The optional second output is the real double filter actually used."] }
,
 BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["Supported providers may perform typed preparation and filtering while preserving owner, device, shape and precision. Before provider effects, unavailable routes may gather through the exact owner; unphased hook failures and malformed provider results are terminal.",
 "Typed integer signal X is rejected. Typed structural/filter controls are independently gated RunMat extensions with exact conversion requirements."] }
],

 examples: &[BuiltinExample {
 id: "example-1",
 title: "Double the sample rate",
 program: "x = [1 2 3];\ny = resample(x,2,1);\nsize(y)",
 display_output: Some("ans = [1 6]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(y),[1 6])); assert(strcmp(class(y),'double'));" }
 }
,
 BuiltinExample {
 id: "example-2",
 title: "Inspect the used filter",
 program: "x = [1 2 3 4];\n[y,b] = resample(x,1,2);\nsize(y)",
 display_output: Some("ans = [1 2]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(y),[1 2])); assert(isrow(b)); assert(strcmp(class(b),'double'));" }
 }
],
 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq {
 question: "How is the output length chosen?",
 answer: "The selected input extent is multiplied by P, divided by Q, and rounded upward." }
,
 BuiltinDocumentationFaq {
 question: "What does the second output contain?",
 answer: "The FIR coefficients actually used." }
,
 BuiltinDocumentationFaq {
 question: "Does resample preserve single precision?",
 answer: "Yes for single X; the filter output remains double." }
,
 BuiltinDocumentationFaq {
 question: "Are nonuniform time vectors and timetables supported?",
 answer: "Not by these uniform rational-rate forms; use a time-aware interpolation or timetable workflow." }
],

 links: &[BuiltinDocumentationLink {
 label: "upsample",
 target: BuiltinDocumentationLinkTarget::Builtin("upsample") }
,
 BuiltinDocumentationLink {
 label: "downsample",
 target: BuiltinDocumentationLinkTarget::Builtin("downsample") }
,
 BuiltinDocumentationLink {
 label: "filter",
 target: BuiltinDocumentationLinkTarget::Builtin("filter") }
,
 BuiltinDocumentationLink {
 label: "interp1",
 target: BuiltinDocumentationLinkTarget::Builtin("interp1") }
,
 BuiltinDocumentationLink {
 label: "Source",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sample_rate.rs") }
],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink {
 label: "Rational conversion and provider preparation",
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
 label: "Host, precision, filter and provider semantics",
 location: "builtins::math::signal::sample_rate::tests" }
],
 notes: &[] }
,
 introduced: None,
 status: None,

}
;
