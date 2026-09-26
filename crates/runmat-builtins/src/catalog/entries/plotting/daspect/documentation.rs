use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("daspect"),
slug:Some("daspect"),
summary:"Query or set axes data aspect ratio.",
description:"`daspect` controls the data aspect ratio for the current or specified axes. RunMat supports ratio query, positive three-element ratio assignment, mode query, `auto`/`manual` mode assignment, and the corresponding `DataAspectRatio` and `DataAspectRatioMode` axes properties.",
keywords:&["daspect",
"data aspect ratio",
"plotting",
"axes"],
related:&["axis",
"xscale",
"yscale"],
sections:&[BuiltinDocumentationSection{
heading:"Behavior and execution",
paragraphs:&[
"`daspect([dx dy dz])` sets `DataAspectRatioMode` to `manual`.",

"Ratio and mode setter forms are statement-only and do not return a value; assigning an output from a setter is rejected.",

"MATLAB documents all eight integer classes as well as single and double ratio vectors. RunMat validates integer shape and positivity from authoritative storage, then crosses once into the double-valued graphics property.",

"Logical, complex, nonvector, and GPU-resident ratio inputs are rejected without provider access.",

"RunMat's numeric or typed-integer encoded axes handles are available only through the `daspect-numeric-axes-handle-alias` extension; the public MATLAB contract uses an Axes object.",

"`daspect('auto')` and `set(gca,'DataAspectRatioMode','auto')` preserve the stored ratio but return display bounds to automatic aspect fitting.",

"RunMat applies manual X/Y data aspect ratios to 2-D display bounds and X/Y/Z data aspect ratios to 3-D camera-fit bounds.",

"RunMat currently represents axes with numeric aliases; strict compatibility mode therefore rejects explicit numeric target aliases.",

"`daspect` reads and writes axes metadata. Manual data aspect ratios are applied by the plotting display-bounds and camera-fit paths; numeric plot buffers are not gathered or transformed.",

"`daspect` is axes metadata only. It does not gather, upload, or mutate numeric GPU buffers; rendering applies the stored aspect metadata through the normal plot camera/bounds path.",

"Queries produce a host double 1-by-3 ratio or a host string mode. More than one output rejects before state access. Unknown controls remain dynamic in inference, rather than claiming a ratio from argument shape alone.",

"All runtime paths share the existing axes metadata and display-bounds/camera-fit owners. Shared figure updates may invoke host observers. Numeric aliases reference local graphics state, not transferable graphics ownership.",

]}],
examples:&[
BuiltinExample{
id:"example-1",
title:"Set and query the data aspect ratio",
program:"plot(0:10, sin(0:10));\ndaspect([1 2 1]);\nget(gca, 'DataAspectRatio')",
display_output:Some("ans =\n     1     2     1"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(isequal(daspect(),[1 2 1])); assert(strcmp(daspect('mode'),'manual'));"}},

BuiltinExample{
id:"example-2",
title:"Target a specific subplot using a RunMat axes alias",
program:"ax = subplot(1,2,1);\nplot(ax, 1:4, [1 4 9 16]);\ndaspect(ax, [1 1 1]);\ndaspect(ax, 'mode')",
display_output:Some("ans =\n    'manual'"),
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(isequal(daspect(ax),[1 1 1])); assert(strcmp(daspect(ax,'mode'),'manual'));"}},

],
example_exemption:None,
faqs:&[],
links:&[
BuiltinDocumentationLink{
label:"axis",
target:BuiltinDocumentationLinkTarget::Builtin("axis")},

BuiltinDocumentationLink{
label:"xscale",
target:BuiltinDocumentationLinkTarget::Builtin("xscale")},

BuiltinDocumentationLink{
label:"yscale",
target:BuiltinDocumentationLinkTarget::Builtin("yscale")},

BuiltinDocumentationLink{
label:"Implementation",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/daspect.rs")}],
media:&[],
evidence:BuiltinDocumentationEvidence{
implementation:&[],
verification:&[BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Ratio, mode, integer and provider rejection tests",
location:"builtins::plotting::daspect::tests"}],
notes:&[
"Interpreter, JIT and AOT use the typed runtime binding.",

"Headless state tests establish metadata semantics only. These typed BrowserGraphics examples require actual figure rendering; Desktop presentation and native engine execution remain distinct integration witnesses.",

"Legacy notes, renderer/bounds behavior, signatures, links and both examples retained. Explicit numeric target example correctly declares RunMat compatibility. Numeric storage remains authoritative until its existing conversion to double graphics metadata."]},
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable)};
