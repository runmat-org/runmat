use crate::*;
pub(super)const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("histc"),
slug:Some("histc"),
summary:"Count values in legacy MATLAB histogram bins.",
description:"`histc(x, binranges)` counts values in legacy MATLAB histogram bins. Each bin includes its left endpoint and excludes its right endpoint, while the final bin counts values exactly equal to the last bin range value.",
keywords:&[
"histc",
"histogram",
"legacy histogram",
"bin counts",
"bin indices"]
,
related:&[
"histcounts",
"histogram"]
,
sections:&[
BuiltinDocumentationSection{
heading:"Behavior",
paragraphs:&[
"Vector inputs return counts with the same orientation as vector `binranges`.",
"Matrix inputs operate down columns by default; `histc(x, binranges, dim)` operates along the requested dimension.",
"`[bincounts, ind] = histc(...)` returns one-based bin indices with the same shape as `x`; values outside the ranges or equal to `NaN` receive index zero.",
"Empty `binranges` return empty count and index arrays.",
"`histc` supports at most two requested outputs.",
"`binranges` must be monotonically nondecreasing and must not contain `NaN`; repeated edges are allowed.",
"Complex inputs and complex bin ranges use their real parts, matching MATLAB's legacy behavior.",
"Matrix `binranges` are interpreted column-by-column when the number of columns matches the number of operated slices."]
}
,
BuiltinDocumentationSection{
heading:"Classes and execution",
paragraphs:&[
"Counts and indices are host double arrays. Integer inputs enter the existing floating histogram comparison domain; wide integers can round at that boundary.",
"Logical inputs, int64/uint64 inputs, and resident inputs are gated RunMat extensions. Complex values use their real components.",
"Resident values are gathered through the owning provider. This operation is a materializing fusion boundary; it has no executable reduction-fusion kernel. Dimensions must be positive integers no larger than 32."]
}
]
,
examples:&[
BuiltinExample{
id:"example-1",
title:"Count ages in legacy bins",
program:"ages = [3 12 24 15 5 74 23 54 31 23 64 75];\nbinranges = [0 10 25 50 75];\n[bincounts, ind] = histc(ages, binranges)",
display_output:Some("bincounts = [2 5 1 3 1]\nind = [1 2 2 2 1 4 2 4 3 2 4 5]"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isequal(bincounts,[2 5 1 3 1])); assert(isequal(ind,[1 2 2 2 1 4 2 4 3 2 4 5]));"}
}
,
BuiltinExample{
id:"example-2",
title:"Operate along rows",
program:"x = [0 1 2; 2 3 4];\nbincounts = histc(x, [0 2 4], 2)",
display_output:Some("bincounts = [2 1 0; 0 2 1]"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isequal(bincounts,[2 1 0;0 2 1]));"}
}
,
BuiltinExample{
id:"example-3",
title:"Values outside the ranges",
program:"[bincounts, ind] = histc([-1 NaN 0.5 2 3], [0 1 2])",
display_output:Some("bincounts = [1 0 1]\nind = [0 0 1 3 0]"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(isequal(bincounts,[1 0 1])); assert(isequal(ind,[0 0 1 3 0]));"}
}
]
,
example_exemption:None,
faqs:&[
BuiltinDocumentationFaq{
question:"Should new code use histc?",
answer:"No. MATLAB marks `histc` as not recommended; use `histcounts` for new code. RunMat supports `histc` for backwards compatibility with existing scripts."}
,
BuiltinDocumentationFaq{
question:"How does histc differ from histcounts?",
answer:"`histc` uses legacy bin ranges and includes a final equality-only bin at the last range value. `histcounts` returns modern edge-based bins and supports normalization and bin-selection options."}
,
BuiltinDocumentationFaq{
question:"Does histc include out-of-range values?",
answer:"No. Values below the first range, above the last range, or equal to `NaN` do not contribute to counts and receive bin index zero."}
]
,
links:&[
BuiltinDocumentationLink{
label:"histcounts",
target:BuiltinDocumentationLinkTarget::Builtin("histcounts")}
,
BuiltinDocumentationLink{
label:"histogram",
target:BuiltinDocumentationLinkTarget::Builtin("histogram")}
,
BuiltinDocumentationLink{
label:"Source",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/hist/histc/mod.rs")}
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
label:"Legacy histogram semantics",
location:"builtins::stats::hist::histc::tests"}
]
,
notes:&[
"All legacy descriptive fields, examples, FAQ and links retained. Second example display corrected: each row counts [0,2), [2,4), and equality with 4; executable assertion matches runtime regression. Legacy fusion.reduction=true described a histogram, not an implemented reduction kernel; typed boundary is authoritative. Null requires_feature/integration omitted; source link follows modular implementation."]
}
,
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable)}
;
