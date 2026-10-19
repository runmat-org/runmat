use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("axes"),
slug:Some("axes"),
summary:"Create axes or make an existing axes current.",
description:"`axes` creates a new axes in the current figure, creates axes in a target figure, or selects an existing axes handle as current. Property/value pairs are applied during construction or selection.",
keywords:&["axes",
"graphics",
"plotting",
"current axes"],
related:&["gca",
"gcf",
"figure",
"get",
"set",
"subplot"],
sections:&[BuiltinDocumentationSection{
heading:"Behavior",
paragraphs:&[
"`axes()` creates a new axes in the current figure and returns its numeric axes handle.",

"`axes(fig)` creates a new axes parented to the specified figure handle.",

"`axes(ax)` makes an existing axes handle current.",

"`axes('Parent', fig, ...)` creates axes in the requested figure and applies the remaining axes properties.",

"`axes(ax, property, value, ...)` selects an existing axes and applies the supplied properties.",

"`Position` and `Units` round-trip through `get`, `set`, and dot-property access. Explicit `Position` values participate in renderer viewport layout without disabling per-axes fast paths.",

"Supported numeric ruler, aspect-ratio, layout, font-size, and tick-label-rotation properties accept all eight integer classes. Authoritative integer storage is validated before one deliberate conversion into host `double` graphics state; the returned axes handle is opaque graphics state rather than integer data.",

"GPU-resident property values reject before provider access because `axes` is a host graphics-state operation and exposes no interactive GPU-array input surface.",

"Zero requested outputs still creates or selects axes. More than one output rejects before state changes. Property/value pair syntax is checked before changing the current selection.",

"Properties are applied in order by the shared graphics property implementation. Property errors are not transactional: axes creation or selection and earlier property updates may remain after a later property value is rejected.",

"Numeric handle aliases refer to local figure and axes state. Copying the number to a worker or foreign runtime does not transfer graphics ownership.",

]},
BuiltinDocumentationSection{
heading:"Placement",
paragraphs:&["`axes` updates host-side plotting state and viewport metadata. Numeric plot data can still use the existing GPU rendering fast paths.",
"`axes` returns host-side handles and does not change array GPU residency.",
"`axes` performs no provider dispatch; it updates plotting registry state and renderer axes metadata. Resident property values reject rather than gathering."]}],
examples:&[
BuiltinExample{
id:"example-1",
title:"Create positioned axes",
program:"ax = axes('Position', [0.2 0.2 0.6 0.6]);\nplot(ax, 1:10);",
display_output:Some("% ax is a numeric axes handle"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(strcmp(get(ax,'Type'),'axes')); assert(isequal(get(ax,'Position'),[0.2 0.2 0.6 0.6]));"}},

BuiltinExample{
id:"example-2",
title:"Create axes in a specific figure",
program:"f = figure(3);\nax = axes('Parent', f, 'Units', 'pixels', 'Position', [40 40 320 220]);",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(get(ax,'Parent')==f); assert(strcmp(get(ax,'Units'),'pixels')); assert(isequal(get(ax,'Position'),[40 40 320 220]));"}},

BuiltinExample{
id:"example-3",
title:"Select an existing axes",
program:"ax1 = axes();\nax2 = axes();\naxes(ax1);",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(ax1~=ax2); assert(gca()==ax1);"}},

],
example_exemption:None,
faqs:&[
BuiltinDocumentationFaq{
question:"Does `axes` replace `gca`?",
answer:"No. Use `axes` when creating or selecting axes. Use `gca` when reading the current axes handle."},

BuiltinDocumentationFaq{
question:"Does an explicit axes `Position` force a slow renderer path?",
answer:"No. Positioned axes are converted into the same per-axes viewport rectangles used by subplot rendering, preserving direct viewport mapping where it is otherwise applicable."},

BuiltinDocumentationFaq{
question:"Does `axes` run numeric kernels?",
answer:"No. It mutates plotting state only."},

],
links:&[
BuiltinDocumentationLink{
label:"gca",
target:BuiltinDocumentationLinkTarget::Builtin("gca")},

BuiltinDocumentationLink{
label:"gcf",
target:BuiltinDocumentationLinkTarget::Builtin("gcf")},

BuiltinDocumentationLink{
label:"figure",
target:BuiltinDocumentationLinkTarget::Builtin("figure")},

BuiltinDocumentationLink{
label:"get",
target:BuiltinDocumentationLinkTarget::Builtin("get")},

BuiltinDocumentationLink{
label:"set",
target:BuiltinDocumentationLinkTarget::Builtin("set")},

BuiltinDocumentationLink{
label:"subplot",
target:BuiltinDocumentationLinkTarget::Builtin("subplot")},

BuiltinDocumentationLink{
label:"Implementation",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/axes.rs")}],
media:&[],
evidence:BuiltinDocumentationEvidence{
implementation:&[],
verification:&[BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Axes identity and property admission",
location:"builtins::plotting::axes::tests"},
BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::IntegrationTest,
label:"VM graphics calls",
location:"crates/runmat-vm/tests/plotting.rs"}],
notes:&["All substantive legacy prose, FAQs, examples and links retained. Host controls had no legacy automatic promotion or gather declaration. Interpreter/JIT/AOT route through the typed runtime binding; actual native and Browser/Desktop rendering remain integration witnesses, not inferred from state-only unit tests.",
"Shared axes property admission remains ordered and nontransactional. Transactional validation/publication is an L20/L21 graphics-property owner obligation; this migration does not introduce a second property parser or claim rollback."]},
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable)};
