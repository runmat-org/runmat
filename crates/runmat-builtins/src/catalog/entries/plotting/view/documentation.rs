use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("view"),
slug:Some("view"),
summary:"Set or query 3-D camera view angles with MATLAB-compatible `view` call forms.",
description:"`view` reads or updates azimuth/elevation camera orientation for current or specified axes. It supports MATLAB-compatible forms such as `view(2)`, `view(3)`, `view([az el])`, and `view(az, el)`.",
keywords:&["view",
"3-D camera",
"azimuth elevation",
"matlab view",
"rotate 3-D plot"],
related:&["plot3",
"surf",
"get",
"set"],
sections:&[BuiltinDocumentationSection{
heading:"Behavior and execution",
paragraphs:&[
"`view` with no angle arguments queries the current `[az el]` pair.",

"`view(2)` selects the standard top-down 2-D camera and `view(3)` restores the standard 3-D camera preset.",

"Explicit azimuth/elevation values can be passed as two scalars or as a two-element vector.",

"View state is subplot-local and does not leak across different axes in a subplot grid.",

"Axes handles can be passed explicitly so camera changes target a specific subplot rather than whichever axes happens to be current.",

"Typed-integer angles and numeric axes aliases are independently gated RunMat extensions. Integer angles must be exactly representable in the renderer's single-precision camera state before any plot mutation.",

"Three-element line-of-sight vectors and arrays of axes handles are not implemented. The documented two-output angle query is supported.",

"With two explicitly requested outputs, [az,el] = view(...) returns scalar doubles for the current or assigned camera angles. With one explicitly requested output, the RunMat-only vector-output extension returns [az el]; strict MATLAB mode rejects this extension rather than claiming a MATLAB view-matrix result. Zero outputs still query or update the selected axes.",

"The existing shared camera state stores angles in single precision. Floating inputs must remain finite across that renderer boundary; typed integers must round-trip exactly. Query outputs are double representations of the stored values.",

"These graphics metadata controls require host numeric values: no implicit device download or numeric acceleration is declared. Handles refer to local axes state, not transferable worker/foreign graphics ownership. Output admission and numeric parsing precede default-axes creation.",

"Figure observers are notified through the shared axes-state owner. ",

]}],
examples:&[
BuiltinExample{
id:"example-1",
title:"Set an explicit 3-D camera angle",
program:"[X, Y] = meshgrid(linspace(-2, 2, 60), linspace(-2, 2, 60));\nZ = sin(X.^2 + Y.^2);\nsurf(X, Y, Z);\nview(45, 30);",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"[az,el]=view(); assert(az==45); assert(el==30);"}},

BuiltinExample{
id:"example-2",
title:"Use MATLAB-style view presets",
program:"[X, Y] = meshgrid(linspace(-3, 3, 40), linspace(-3, 3, 40));\nZ = sin(X) .* cos(Y);\nsurf(X, Y, Z);\nview(2);\nview(3);",
display_output:None,
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"[az,el]=view(); assert(az==-37.5); assert(el==30);"}},

BuiltinExample{
id:"example-3",
title:"Query a subplot-local camera",
program:"ax = subplot(1, 2, 2);\nplot3(cos(0:0.1:5), sin(0:0.1:5), 0:0.1:5);\nview(ax, [60 20]);\nget(ax, 'View')",
display_output:Some("% Returns a 1x2 vector [60 20]"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"[az,el]=view(ax); assert(az==60); assert(el==20); assert(isequal(get(ax,'View'),[60 20]));"}},

],
example_exemption:None,
faqs:&[
BuiltinDocumentationFaq{
question:"What do azimuth and elevation mean in view(az, el)?",
answer:"Azimuth is the horizontal rotation around the z-axis (0° looks along the y-axis toward the origin). Elevation is the angle above the x-y plane (0° is level, 90° is directly overhead). `view(45, 30)` gives the classic isometric-ish perspective for 3-D surfaces."},

BuiltinDocumentationFaq{
question:"How do I reset to the default 2-D or 3-D view?",
answer:"`view(2)` sets the camera to top-down (azimuth 0, elevation 90) — useful for looking at a surface as a heatmap. `view(3)` restores the default 3-D camera at `[-37.5, 30]`, which is the standard MATLAB 3-D preset."},

BuiltinDocumentationFaq{
question:"Can I set different view angles per subplot?",
answer:"Yes. View state is subplot-local. Pass the axes handle explicitly to target a specific panel.\n\n```matlab\nax1 = subplot(1, 2, 1); surf(X, Y, Z); view(ax1, 0, 90);\nax2 = subplot(1, 2, 2); surf(X, Y, Z); view(ax2, 45, 30);\n```"},

],
links:&[
BuiltinDocumentationLink{
label:"plot3",
target:BuiltinDocumentationLinkTarget::Builtin("plot3")},

BuiltinDocumentationLink{
label:"scatter3",
target:BuiltinDocumentationLinkTarget::Builtin("scatter3")},

BuiltinDocumentationLink{
label:"surf",
target:BuiltinDocumentationLinkTarget::Builtin("surf")},

BuiltinDocumentationLink{
label:"zlabel",
target:BuiltinDocumentationLinkTarget::Builtin("zlabel")},

BuiltinDocumentationLink{
label:"get",
target:BuiltinDocumentationLinkTarget::Builtin("get")},

BuiltinDocumentationLink{
label:"set",
target:BuiltinDocumentationLinkTarget::Builtin("set")},

BuiltinDocumentationLink{
label:"Choosing the right plot type",
target:BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type")},

BuiltinDocumentationLink{
label:"Styling plots and axes",
target:BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes")},

BuiltinDocumentationLink{
label:"Plot replay and export",
target:BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export")},

BuiltinDocumentationLink{
label:"Complete plotting guide",
target:BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide")},

BuiltinDocumentationLink{
label:"Implementation",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/view.rs")},
BuiltinDocumentationLink{
label:"MATLAB view contract",
target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/view.html")}],
media:&[],
evidence:BuiltinDocumentationEvidence{
implementation:&[],
verification:&[BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Camera-state and output tests",
location:"builtins::plotting::view::tests"}],
notes:&[
"Interpreter and compiled execution share the typed runtime binding.",

"Headless state tests do not prove UI presentation; BrowserGraphics examples require actual rendering, with Desktop and native engine witnesses qualified separately.",

"Legacy prose, FAQs, links and all three examples retained; the formerly unsupported two-output form now adapts existing state. Existing single-vector output is explicitly a RunMat extension. No browser/desktop/native execution claim is inferred from source or headless tests."]},
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable)};
