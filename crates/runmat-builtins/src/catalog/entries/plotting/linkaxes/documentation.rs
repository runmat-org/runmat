use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("linkaxes"),
slug:Some("linkaxes"),
summary:"Link axes limits across multiple axes.",
description:"`linkaxes(ax)` links the x- and y-axis limits for a vector of axes handles. `linkaxes(ax, option)` links only the requested axis limits or removes existing links.",
keywords:&["linkaxes",
"linked axes",
"xlim",
"ylim",
"plotting"],
related:&["axes",
"subplot",
"xlim",
"ylim"],
sections:&[BuiltinDocumentationSection{
heading:"Behavior",
paragraphs:&[
"`linkaxes(ax)` is equivalent to `linkaxes(ax, 'xy')`.",

"`linkaxes(ax, 'x')` links X limits while leaving Y limits independent.",

"`linkaxes(ax, 'y')` links Y limits while leaving X limits independent.",

"`linkaxes(ax, 'xy')` links both X and Y limits.",

"`linkaxes(ax, 'off')` removes the supplied axes from any existing linked-axes groups.",

"When a new link is created, linked axes are synchronized to limits that encompass the current limits of every supplied axes handle for the linked axes.",

"Axis-specific links can overlap. For example, one axes can stay linked to another axes on X limits while also being linked to a different axes on Y limits.",

"The numeric handle array may have any rank. Duplicates are deduplicated by the shared link owner; an empty array is accepted. Fewer than two distinct handles creates no new link, while the requested link-axis memberships of supplied handles are removed.",

"This command has no outputs. Explicit output requests fail before linking or unlinking; implicit and zero-output statements execute normally. All supplied handles and the mode are validated before linked-limit mutation.",

"Integer handle storage is decoded exactly and must represent valid existing axes. Resident or sparse handles are unsupported; no automatic upload or download is introduced. Linked state and handle identities are local to the graphics host, not implicitly transferable to workers or foreign runtimes.",

"Host figure updates can invoke observers.",

]},
BuiltinDocumentationSection{
heading:"Placement",
paragraphs:&["`linkaxes` updates host-side graphics state only and does not execute numeric GPU kernels.",
"`linkaxes` works with host-side graphics handles and does not move arrays between host and GPU memory.",
"`linkaxes` performs no provider dispatch and has no GPU kernel implementation."]}],
examples:&[
BuiltinExample{
id:"example-1",
title:"Link x limits for two subplots",
program:"ax1 = subplot(2,1,1);\nax2 = subplot(2,1,2);\nlinkaxes([ax1 ax2], 'x');\nset(ax1, 'XLim', [0 10]);\nget(ax2, 'XLim')",
display_output:Some("ans = [0 10]"),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(isequal(get(ax2,'XLim'),[0 10]));"}},

BuiltinExample{
id:"example-2",
title:"Remove a link",
program:"ax1=subplot(2,1,1);\nax2=subplot(2,1,2);\nset(ax1,'XLim',[0 1]);\nset(ax2,'XLim',[0 1]);\nlinkaxes([ax1 ax2],'x');\nlinkaxes([ax1 ax2],'off');\nset(ax1,'XLim',[2 3]);",
display_output:Some(""),
compatibility:BuiltinExampleCompatibility::Matlab,
harness:BuiltinExampleHarness::BrowserGraphics,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Figure{
minimum_figures:1,
assertions:"assert(isequal(get(ax1,'XLim'),[2 3])); assert(isequal(get(ax2,'XLim'),[0 1]));"}},

],
example_exemption:None,
faqs:&[
BuiltinDocumentationFaq{
question:"Does `linkaxes` create axes?",
answer:"No. It requires existing axes handles such as values returned from `subplot`, `axes`, or `gca`."},

BuiltinDocumentationFaq{
question:"Does linking affect future `set(ax, 'XLim', ...)` calls?",
answer:"Yes. Linked limits propagate through RunMat's central axes-limit state path, including `xlim`, `ylim`, `set`, and zoom-related limit updates. Unlinked axes continue to use the direct limit-update path."},

],
links:&[
BuiltinDocumentationLink{
label:"axes",
target:BuiltinDocumentationLinkTarget::Builtin("axes")},

BuiltinDocumentationLink{
label:"subplot",
target:BuiltinDocumentationLinkTarget::Builtin("subplot")},

BuiltinDocumentationLink{
label:"xlim",
target:BuiltinDocumentationLinkTarget::Builtin("xlim")},

BuiltinDocumentationLink{
label:"ylim",
target:BuiltinDocumentationLinkTarget::Builtin("ylim")},

BuiltinDocumentationLink{
label:"Implementation",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/linkaxes.rs")}],
media:&[],
evidence:BuiltinDocumentationEvidence{
implementation:&[],
verification:&[BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Shared linked-limit behavior and admission",
location:"builtins::plotting::linkaxes::tests"},
BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::IntegrationTest,
label:"VM linked axes",
location:"runmat-vm plotting::linkaxes_propagates_limits_and_supports_off_mode"}],
notes:&[
"Interpreter, JIT and AOT use the same typed binding and shared linked-limit owner. State-only unit tests do not prove visible rendering; actual BrowserGraphics and Desktop presentation remain integration gates.",

"All legacy prose, FAQs and links preserved; unlink example now supplies its own prerequisite axes/link and semantic assertions. Legacy empty string output was an implementation carrier, not a documented public result; typed adaptation now returns an empty output sequence."]},
introduced:None,
status:Some(BuiltinDocumentationStatus::Stable)};
