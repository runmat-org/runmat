use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("hold"),
 slug: Some("hold"),
 summary: "Control plot replacement versus appending in MATLAB and RunMat.",
 description: "`hold` controls whether new plotting commands append to the current axes or replace existing plot objects. Per-axes hold state and command forms follow MATLAB semantics.",
 keywords: &["hold",
"hold on",
"hold off",
"plotting",
"append plots"],
 related: &["plot",
"cla",
"clf",
"subplot"],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"`hold()` toggles the current hold state.",

"`hold('on')` enables appending to the current axes.",

"`hold('off')` disables appending so later plots replace the axes contents.",

"`hold('all')` is currently treated the same as `hold('on')`.",

"Logical and numeric scalar inputs are accepted: zero means off and one means on.",

"An axes handle can be supplied before the mode to update that axes.",

]},

BuiltinDocumentationSection {
 heading: "Placement and execution",
 paragraphs: &[
"`hold` changes plotting state only. It does not execute numeric GPU kernels.",

"`hold` only mutates plotting state and does not affect array residency.",

"`hold` performs no provider dispatch; it updates per-axes plotting state in the host-side registry.",

"Numeric controls must already be host values. No legacy automatic promotion or gathering was declared for this identity. Handle numbers refer to local graphics state; copying a handle does not transfer graphics ownership to workers or foreign runtimes.",

"Explicitly requested outputs reject before state changes; implicit statement calls and zero outputs execute normally. The operation may notify shared figure observers when an explicit target becomes current, but does not itself prove pixel presentation.",

"Empty character or string text also toggles the state. The literal word toggle is not accepted. Numeric states are validated exactly as zero or one; complex, sparse, resident arrays and dense logical-array representations are unsupported.",

]},

],
examples: &[
BuiltinExample {
 id: "example-1",
 title: "Append a second line to the same axes",
 program: "x = 0:0.1:1;\nh1 = plot(x,x);\nhold on;\nh2 = plot(x,x.^2);",
 display_output: Some("% both lines remain visible on the current axes"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure {
 minimum_figures: 1,
 assertions: "assert(h1 ~= h2); assert(isequal(get(h1,'YData'),x)); assert(isequal(get(h2,'YData'),x.^2));" } },

BuiltinExample {
 id: "example-2",
 title: "Disable appending again",
 program: "hold off;\nh1 = plot(1:5,[1 2 3 4 5]);\nh2 = plot(1:5,[5 4 3 2 1]);",
 display_output: Some("% the new plot replaces the previous axes contents"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure {
 minimum_figures: 1,
 assertions: "assert(isequal(get(h2,'YData'),[5 4 3 2 1]));" } },

BuiltinExample {
 id: "example-3",
 title: "Toggle hold state",
 program: "hold off;\nhold();\nhold();\nh1 = plot([1 2],[1 2]);\nh2 = plot([1 2],[2 1]);",
 display_output: Some("% each call toggles the current axes hold state"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure {
 minimum_figures: 1,
 assertions: "assert(isequal(get(h2,'YData'),[2 1]));" } },

],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq {
 question: "Is hold state per figure or per axes?",
 answer: "It is tracked per axes, so one subplot can be in hold mode while another is not." },

BuiltinDocumentationFaq {
 question: "What does `hold('all')` do?",
 answer: "RunMat currently treats `'all'` the same as `'on'`." },

BuiltinDocumentationFaq {
 question: "Does hold return the current state?",
 answer: "No. `hold` mutates axes state and has no public output." },

],
 links: &[
BuiltinDocumentationLink {
 label: "plot",
 target: BuiltinDocumentationLinkTarget::Builtin("plot") },

BuiltinDocumentationLink {
 label: "cla",
 target: BuiltinDocumentationLinkTarget::Builtin("cla") },

BuiltinDocumentationLink {
 label: "clf",
 target: BuiltinDocumentationLinkTarget::Builtin("clf") },

BuiltinDocumentationLink {
 label: "subplot",
 target: BuiltinDocumentationLinkTarget::Builtin("subplot") },

BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/hold.rs") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference {
kind: BuiltinEvidenceKind::UnitTest,
label: "Hold state and output admission",
location: "builtins::plotting::hold::tests"}],
 notes: &[
"Interpreter, JIT and AOT share the typed runtime binding.",

"Legacy renderer integration reference: runmat-plot/tests/renderer_tests.rs. State-only tests do not establish Browser or Desktop presentation; typed graphics examples and native engine witnesses require separate integration execution."] },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable) };
