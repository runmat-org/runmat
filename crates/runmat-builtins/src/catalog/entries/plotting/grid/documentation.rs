use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("grid"),
 slug: Some("grid"),
 summary: "Toggle axes grid lines in MATLAB and RunMat plotting.",
 description: "`grid` controls visibility of major and minor axes grid lines on the active axes or subplot. Command forms such as `grid on`, `grid off`, `grid minor`, and toggle behavior follow MATLAB semantics.",
 keywords: &["grid", "grid on", "grid off", "grid minor", "chart grid lines", "matlab grid"],
 related: &["axis", "box", "subplot"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`grid on` enables grid lines on the current axes.",
"`grid off` disables major and minor grid lines on the current axes.",
"`grid minor` toggles minor grid lines without changing the major grid state.",
"Calling `grid` with no argument toggles the major grid state.",
"Grid visibility is subplot-local and does not spill into other axes automatically.",
"An assigned logical state/success result is a RunMat extension. MATLAB compatibility mode allows command calls but rejects an explicitly requested result before changing state."
] },
BuiltinDocumentationSection { heading: "Placement and display", paragraphs: &[
"These commands update shared graphics state and may notify figure observers. Numeric array acceleration remains with the original plotting/rendering pipeline; the commands do not launch array-provider kernels. Graphics display requires a browser graphics host or a desktop graphics host.",
"Control arguments must already be host values. Unsupported resident controls reject without provider downloads; existing resident plot buffers stay owned by their plotting/rendering paths. Invalid syntax and excess output requests are checked before creating or changing graphics state."
] }
 ],
 examples: &[
BuiltinExample {
 id: "example-1",
 title: "Enable grid lines on a simple line chart",
 program: "plot(1:5, [1 4 2 5 3]);\ngrid on;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(get(gca,'Grid'));" },
},
BuiltinExample {
 id: "example-2",
 title: "Toggle minor grid lines on the current axes",
 program: "plot(1:10,sin(1:10)); grid off; grid minor;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(~get(gca,'Grid')); assert(get(gca,'MinorGrid'));" },
},
BuiltinExample {
 id: "example-3",
 title: "Toggle grid lines in one subplot only",
 program: "ax1 = subplot(1,2,1); plot(1:5,1:5); grid off;\nax2 = subplot(1,2,2); plot(1:5,[5 4 3 2 1]); grid on;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(~get(ax1,'Grid')); assert(get(ax2,'Grid'));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Does RunMat support minor grid lines?", answer: "Yes. Use `grid minor` or `grid('minor')` to toggle minor grid lines on the active axes. `grid off` disables both major and minor grid lines." },
BuiltinDocumentationFaq { question: "Can I style the grid line color or width?", answer: "RunMat currently exposes major and minor grid visibility through axes state. GridColor and grid line-width setters are not supported by this property path; `get(gca)` lists the properties that are available." },
BuiltinDocumentationFaq { question: "Does grid state carry across subplots?", answer: "No. Grid visibility is subplot-local. Calling `grid on` in one panel doesn't turn it on in others — you need to enable it explicitly in each subplot where you want grid lines." }
 ],
 links: &[
BuiltinDocumentationLink { label: "axis", target: BuiltinDocumentationLinkTarget::Builtin("axis") },
BuiltinDocumentationLink { label: "box", target: BuiltinDocumentationLinkTarget::Builtin("box") },
BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/grid.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "grid state, output and validation tests", location: "builtins::plotting::cmds::grid::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
