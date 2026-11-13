use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

 example_exemption: None,

 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("magic"),
 slug: Some("magic"),

 summary: "Generate magic squares in MATLAB and RunMat.",
 description: concat!(
"`magic(n)` returns a host double n-by-n matrix. The first real element ",

"selects the order; floating-point values are floored and integer ",

"values are decoded exactly. Odd, doubly-even, and singly-even orders ",

"use separate constructions.",

),
 keywords: &["magic",
"magic square",
"matrix",
"array"],
 related: &[],

 sections: &[BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[concat!(
"`magic(n)` accepts documented numeric and character classes, selects ",

"the first real element, floors it, and requires the resulting order to ",

"be non-negative and representable.",

),
"`magic(1)` returns `1`, while `magic(0)` returns an empty `0 x 0` matrix.",
concat!(
"`magic(2)` returns the documented degenerate matrix `[1 3;4 2]`, which ",

"is not a true magic square.",

),
concat!(
"Odd, doubly-even (`n % 4 == 0`), and singly-even (`n % 4 == 2`) orders ",

"each use MATLAB's standard construction so results match MATLAB ",

"outputs.",

)] },
BuiltinDocumentationSection {
 heading: "GPU residency and provider behavior",
 paragraphs: &[concat!(
"`magic` always runs on the host CPU today. If you need the result on ",

"the GPU, wrap the call in `gpuArray(...)`.",

),
concat!(
"RunMat transparently gathers automatically resident inputs before ",

"evaluating `magic` and returns a host tensor. Explicit gpuArray input ",

"is rejected because the documented API does not support it.",

),
concat!(
"Computed on the host CPU. Automatically resident inputs gather ",

"transparently; explicit gpuArray input is unsupported.",

)] }],

 examples: &[BuiltinExample {
 id: "example-1",
 title: "A 3x3 magic square",
 program: "M = magic(3)",
 display_output: Some("M =\n     8     1     6\n     3     5     7\n     4     9     2"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(M, [8 1 6; 3 5 7; 4 9 2])); assert(strcmp(class(M), \"double\"));" } },
BuiltinExample {
 id: "example-2",
 title: "A 4x4 magic square",
 program: "M = magic(4)",
 display_output: Some(concat!(
"M =\n    16     2     3    13\n     5    11    10     8\n     9     7     ",

"6    12\n     4    14    15     1",

)),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(M, [16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]));" } },
BuiltinExample {
 id: "example-3",
 title: "Use magic squares as structured input",
 program: "A = magic(5);\nresult = A * A'",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: concat!(
"assert(isequal(size(result), [5 5])); assert(isequal(result, ",

"result')); assert(all(sum(A) == 65));",

) } }],

 faqs: &[BuiltinDocumentationFaq {
 question: "Is `magic(2)` a true magic square?",
 answer: concat!(
"No. It returns `[1 3;4 2]` for compatibility, even though no true 2x2 ",

"magic square exists.",

) },
BuiltinDocumentationFaq {
 question: "What happens when `n` is an array?",
 answer: concat!(
"The first element controls the order after taking its real part and ",

"flooring it. Use `ones`, `zeros`, or `eye` when you need a size-vector ",

"form.",

) },
BuiltinDocumentationFaq {
 question: "How do I move the result to the GPU?",
 answer: concat!(
"Call `gpuArray(magic(n))` if you need the output on the GPU. RunMat ",

"computes `magic` on the host today.",

) }],

 links: &[BuiltinDocumentationLink {
 label: "ones",
 target: BuiltinDocumentationLinkTarget::Builtin("ones") },
BuiltinDocumentationLink {
 label: "eye",
 target: BuiltinDocumentationLinkTarget::Builtin("eye") },
BuiltinDocumentationLink {
 label: "conv2",
 target: BuiltinDocumentationLinkTarget::Builtin("conv2") },
BuiltinDocumentationLink {
 label: "colon",
 target: BuiltinDocumentationLinkTarget::Builtin("colon") },
BuiltinDocumentationLink {
 label: "false",
 target: BuiltinDocumentationLinkTarget::Builtin("false") },
BuiltinDocumentationLink {
 label: "fill",
 target: BuiltinDocumentationLinkTarget::Builtin("fill") },
BuiltinDocumentationLink {
 label: "linspace",
 target: BuiltinDocumentationLinkTarget::Builtin("linspace") },
BuiltinDocumentationLink {
 label: "logspace",
 target: BuiltinDocumentationLinkTarget::Builtin("logspace") },
BuiltinDocumentationLink {
 label: "meshgrid",
 target: BuiltinDocumentationLinkTarget::Builtin("meshgrid") },
BuiltinDocumentationLink {
 label: "rand",
 target: BuiltinDocumentationLinkTarget::Builtin("rand") },
BuiltinDocumentationLink {
 label: "randi",
 target: BuiltinDocumentationLinkTarget::Builtin("randi") },
BuiltinDocumentationLink {
 label: "randn",
 target: BuiltinDocumentationLinkTarget::Builtin("randn") },
BuiltinDocumentationLink {
 label: "randperm",
 target: BuiltinDocumentationLinkTarget::Builtin("randperm") },
BuiltinDocumentationLink {
 label: "range",
 target: BuiltinDocumentationLinkTarget::Builtin("range") },
BuiltinDocumentationLink {
 label: "true",
 target: BuiltinDocumentationLinkTarget::Builtin("true") },
BuiltinDocumentationLink {
 label: "zeros",
 target: BuiltinDocumentationLinkTarget::Builtin("zeros") },
BuiltinDocumentationLink {
 label: "`crates/runmat-runtime/src/builtins/array/creation/magic.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/s",

"rc/builtins/array/creation/magic.rs",

)) }],
 media: &[],

 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Semantic and boundary tests",
 location: "builtins::array::creation::magic::tests" }],
 notes: &[concat!(
"Legacy prose, syntax, examples, FAQs, and links are retained. Typed ",

"contracts own signatures, integer capabilities, placement and fusion. ",

"Feature and integration fields were null. Native/browser/hardware ",

"evidence is recorded separately.",

)] },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),

};
