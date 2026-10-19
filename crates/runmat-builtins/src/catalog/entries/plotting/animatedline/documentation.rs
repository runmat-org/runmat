use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[
"`animatedline()` creates an empty animated line on the current axes.",

"`animatedline(x,y)` creates a 2-D animated line with initial points.",

"`animatedline(x,y,z)` creates a 3-D animated line when the initial coordinate vectors are nonempty.",

"X, Y, and Z independently accept single, double, and all eight integer classes. Exact integer storage is read authoritatively before coordinates enter RunMat's client-side `f64` graphics domain.",

"A leading axes handle or `Parent` name/value pair routes the animated line to explicit axes.",

"`MaximumNumPoints` accepts every integer class as a positive scalar, or a positive integral single/double value. The default is 1000000; positive `Inf` disables trimming.",

"Integer `MaximumNumPoints` values are validated exactly before conversion to the host point-count type; zero, negative, fractional, nonfinite values other than positive `Inf`, and values beyond host limits reject.",

"Returned handles report `Type` as `animatedline` and support line styling properties such as `Color`, `LineWidth`, `LineStyle`, marker properties for 2-D animated lines, `DisplayName`, and `Visible`.",

] },

BuiltinDocumentationSection { heading: "Options",
 paragraphs: &[
"`Parent` accepts an axes handle.",

"`MaximumNumPoints` accepts a positive integral value in single, double, or any integer class, or positive `Inf`.",

"Common line properties include `Color`, `LineWidth`, `LineStyle`, `DisplayName`, and `Visible`. Marker properties are supported for 2-D animated lines; 3-D animated lines reject marker metadata until RunMat's 3-D line renderer has marker support.",

] },

BuiltinDocumentationSection { heading: "Placement and rendering",
 paragraphs: &[
"`animatedline` is a graphics-state constructor. GPU-resident coordinate inputs are accepted through the plotting input path and gathered into the client-side animated-line state before incremental updates.",

"`animatedline` accepts GPU-resident coordinate inputs but stores animated state as host-side graphics data so subsequent `addpoints`, `get`, and `set` operations have MATLAB-compatible handle semantics.",

"Construction gathers GPU coordinate vectors and resident numeric `MaximumNumPoints` scalars into client-side animated-line state. Integer coordinates cross one explicit floating graphics boundary, while integer point limits are validated exactly before host-size conversion.",

"Rendering still uses RunMat's existing line and line3 rendering paths after points are stored.",

"Resident numeric Color, LineWidth, MarkerSize, MarkerEdgeColor, MarkerFaceColor, Parent, and leading scalar axes handles are materialized at their domain-owned argument positions. Text controls are not sent to a numerical planner. Borrowed resident inputs are not freed by construction.",

] },

BuiltinDocumentationSection { heading: "Execution and effects",
 paragraphs: &[
"This stateful constructor gathers resident coordinate data and numeric point limits through the plotting input service before publishing its animated-line state. It can suspend during gathering, reads presentation configuration, notifies host observers, and uses the existing line/line3 renderer. No automatic upload or numeric fusion kernel is introduced.",

"Interpreter and compiled execution use the same typed runtime binding. Zero requested outputs still performs the operation; more than one output rejects before graphics mutation. Numeric handles identify local graphics state, not portable figure values: workers and foreign adapters must not interpret a copied number as transferred graphics ownership.",

"The existing headless and non-main-thread presentation fallback preserves constructed graphics state even when interactive presentation is unavailable. Browser plotting publishes figure updates; drawnow/pause provide host presentation boundaries. State assertions alone do not prove visible pixels.",

"Coordinate input keeps the existing first-two-extent vector rule: either the first or second extent must be one, and admitted higher dimensions are flattened in storage order. Empty 1-by-0 or 0-by-1 coordinate vectors are accepted; 0-by-0 coordinate arrays are not accepted by this rule. Empty three-coordinate construction starts as 2-D and may later promote through addpoints.",

] },

];
const EXAMPLES: &[BuiltinExample] = &[
BuiltinExample { id: "example-1",
 title: "Create and update a 2-D animated line",
 program: "an = animatedline('Color', 'r', 'MaximumNumPoints', 100);\naddpoints(an, 1:3, [2 4 8]);\nget(an, 'XData')",
 display_output: Some("ans =\n     1     2     3"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: "assert(strcmp(get(an,'Type'),'animatedline')); assert(isequal(get(an,'XData'),[1 2 3])); assert(isequal(get(an,'YData'),[2 4 8]));" } },

BuiltinExample { id: "example-2",
 title: "Create a 3-D animated line",
 program: "an = animatedline();\naddpoints(an, [1 2], [3 4], [5 6]);\nget(an, 'ZData')",
 display_output: Some("ans =\n     5     6"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: "assert(isequal(get(an,'ZData'),[5 6]));" } },

];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("animatedline"),
 slug: Some("animatedline"),
 summary: "Create an animated line graphics object for incremental plotting.",
 description: "`animatedline` creates a graphics handle backed by RunMat's normal 2-D or 3-D line rendering paths. The handle can be updated with `addpoints` and queried or styled with `get` and `set`.",

keywords: &["animatedline",
"addpoints",
"animated line",
"live plot",
"graphics handle",
"matlab plotting"],
 related: &["addpoints",
"line",
"get",
"set"],

sections: SECTIONS,
 examples: EXAMPLES,
 example_exemption: None,

faqs: &[
BuiltinDocumentationFaq { question: "Does animatedline render through a separate animation engine?",
 answer: "No. RunMat stores the animated-line handle as plotting state that points at the existing `LinePlot` or `Line3Plot` object, so updates reuse the established line rendering paths." },

BuiltinDocumentationFaq { question: "What happens when the maximum point count is exceeded?",
 answer: "RunMat drops the oldest coordinates and keeps the newest `MaximumNumPoints` samples, matching MATLAB's compatibility behavior for streaming plots." },

BuiltinDocumentationFaq { question: "Can 3-D animated lines use marker properties?",
 answer: "Not yet. RunMat rejects marker-styled 2-D animated-line handles when callers try to promote them with `addpoints(an,x,y,z)` because the current 3-D line renderer does not store marker metadata." },

],

links: &[
BuiltinDocumentationLink { label: "addpoints",
 target: BuiltinDocumentationLinkTarget::Builtin("addpoints") },

BuiltinDocumentationLink { label: "line",
 target: BuiltinDocumentationLinkTarget::Builtin("line") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") },

BuiltinDocumentationLink { label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/animatedline.rs") },

],
 media: &[],

evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/animatedline.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantic tests",
 location: "builtins::plotting::animatedline::tests" },
 BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "VM graphics dispatch",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &["Legacy signatures, behaviors, options, FAQs, examples, links and GPU notes are preserved or corrected against runtime semantics. Typed catalog placement replaces legacy fusion flags. BrowserGraphics examples require actual graphics-host execution; headless runtime tests do not constitute Browser or Desktop presentation evidence. Native Interpreter/JIT/AOT, real-provider gather, and Desktop presentation qualification remain separate required witnesses."],
 },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),

};
