use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("box"),
 slug: Some("box"),
 summary: "Toggle axes box outlines in MATLAB and RunMat plotting.",
 description: "`box` controls whether the current axes draw their outer frame. It supports `box on`, `box off`, and toggle form with no argument, matching MATLAB and RunMat plotting command behavior.",
 keywords: &["box", "axes box", "box on", "box off", "matlab box"],
 related: &["axis", "grid", "subplot"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`box on` enables the axes frame on the current axes.",
"`box off` disables the axes frame on the current axes.",
"Calling `box` with no argument toggles the current state.",
"Box state is subplot-local and applies to the currently active axes.",
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
 title: "Turn the axes frame on for a chart",
 program: "plot(1:5, [1 4 2 5 3]);\nbox on;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(get(gca,'Box'));" },
},
BuiltinExample {
 id: "example-2",
 title: "Style subplots independently",
 program: "ax1 = subplot(1,2,1); plot(1:5,1:5); box off;\nax2 = subplot(1,2,2); plot(1:5,[5 4 3 2 1]); box on;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(~get(ax1,'Box')); assert(get(ax2,'Box'));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "When should I turn the box off?", answer: "Turn `box off` for a cleaner, more modern chart style — especially for presentations or publications where the top and right frame edges add visual noise without carrying data. Many style guides prefer open axes (left + bottom only) for scatter plots and line charts." },
BuiltinDocumentationFaq { question: "Does box interact with grid or axis settings?", answer: "`box`, `grid`, and `axis` are independent axes-state commands. Turning `box off` removes the outer frame but doesn't touch grid lines or axis limits. You can combine them freely: `box off; grid on; axis tight;` is a common clean-chart recipe." }
 ],
 links: &[
BuiltinDocumentationLink { label: "axis", target: BuiltinDocumentationLinkTarget::Builtin("axis") },
BuiltinDocumentationLink { label: "grid", target: BuiltinDocumentationLinkTarget::Builtin("grid") },
BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/box.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "box state, output and validation tests", location: "builtins::plotting::cmds::box::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
