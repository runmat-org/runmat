use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {

authority:BuiltinDocumentationAuthority::Catalog,
title:Some("gauspuls"),
slug:Some("gauspuls"),
summary:"Generate Gaussian-modulated sinusoidal pulses.",
description:"`gauspuls(T)` evaluates a Gaussian-modulated cosine pulse. Optional `FC`, `BW`, and `BWR` arguments control carrier frequency, fractional bandwidth, and bandwidth reference level. `[YI, YQ, YE] = gauspuls(T, FC, BW, BWR)` also returns quadrature samples and the Gaussian envelope. `gauspuls('cutoff', FC, BW, BWR, TPE)` returns the cutoff time where the envelope reaches `TPE` dB.",
keywords:&[
"gauspuls",
"gaussian pulse",
"pulse train",
"signal processing",
"cutoff"]
,
related:&[
"pulstran",
"rectpuls",
"tripuls"]
,

sections:&[
BuiltinDocumentationSection {
heading:"Behavior",
paragraphs:&[
"Operates element-wise on real numeric scalars, vectors, matrices, and N-D tensors; preserves input shape.",
"One output returns the in-phase cosine component; three requested outputs return in-phase, quadrature, and envelope arrays.",
"Defaults are `FC = 1000`, `BW = 0.5`, and `BWR = -6`.",
"Carrier frequency and bandwidth must be positive finite scalars.",
"Bandwidth reference and cutoff envelope levels must be negative finite scalars.",
"Complex inputs are rejected with a builtin-scoped error."]
}
,
BuiltinDocumentationSection {
heading:"Class and execution",
paragraphs:&[
"Single sample arrays retain single output. Double, logical, and integer samples produce double. Integer sample times and controls must be exactly representable as double; wide inexact values fail instead of rounding.",
"Integer/logical times, integer/logical/single controls, and resident inputs are separately gated RunMat extensions. Empty numeric control arrays select defaults.",
"Resident inputs are downloaded through their owning provider for host evaluation. There is no direct GPU kernel or fusion implementation.",
"Two outputs return in-phase and quadrature samples; three add the envelope. Additional requested outputs currently receive scalar zero padding, including cutoff outputs after the first."]
}
]
,

examples:&[
BuiltinExample {
id:"example-1",
title:"Evaluate a Gaussian pulse",
program:"t = -0.001:0.001:0.001;\ny = gauspuls(t, 1000, 0.5, -6)",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isequal(size(y),size(t))); assert(abs(y(2)-1)<1e-12); assert(abs(y(1)-y(3))<1e-12);"}
}
,
BuiltinExample {
id:"example-2",
title:"Return in-phase, quadrature, and envelope samples",
program:"t = 0:0.00025:0.001;\n[yi, yq, ye] = gauspuls(t, 1000, 0.5, -6)",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isequal(size(yi),size(t))); assert(isequal(size(yq),size(t))); assert(isequal(size(ye),size(t))); assert(max(abs(yi.^2+yq.^2-ye.^2))<1e-12);"}
}
,
BuiltinExample {
id:"example-3",
title:"Compute cutoff time",
program:"tc = gauspuls('cutoff', 1000, 0.5, -6, -60)",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isfinite(tc)); assert(tc>0); [yi,yq,ye]=gauspuls(tc,1000,0.5,-6); assert(abs(ye-0.001)<1e-12);"}
}
]
,
example_exemption:None,

faqs:&[
BuiltinDocumentationFaq{
question:"Can `gauspuls` be used inside `pulstran`?",
answer:"Yes. `pulstran(T, D, 'gauspuls', FC, BW, BWR)` evaluates Gaussian pulses centered at each delay."}
]
,
links:&[
BuiltinDocumentationLink{
label:"pulstran",
target:BuiltinDocumentationLinkTarget::Builtin("pulstran")}
,
BuiltinDocumentationLink{
label:"rectpuls",
target:BuiltinDocumentationLinkTarget::Builtin("rectpuls")}
,
BuiltinDocumentationLink{
label:"tripuls",
target:BuiltinDocumentationLinkTarget::Builtin("tripuls")}
,
BuiltinDocumentationLink{
label:"Source",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/gauspuls/mod.rs")}
]
,
media:&[
]
,
evidence:BuiltinDocumentationEvidence{
implementation:&[
]
,
verification:&[
BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Gaussian pulse semantics and compatibility",
location:"builtins::math::signal::gauspuls::tests"}
]
,
notes:&[
"Legacy title/category/keywords/summary/description/behaviors/examples/FAQ/links retained; GPU and fusion prose reconciled to host gather route. Legacy requires_feature and integration were null. Source link follows the modular implementation. Provider precision includes native single and double; hardware and native execution require separate integrated verification."]
}
,
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable),

}
;
