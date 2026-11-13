use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("hidden"),
 slug: Some("hidden"),
 summary: "Control axes hidden-line-removal state for surface and mesh workflows.",
 description: "`hidden` is MATLAB's legacy command-form helper for toggling hidden-line removal on axes. RunMat stores the state on the target axes as the `HiddenLineRemoval` property and exposes it through `get`/`set`, so older plotting scripts using `hidden on` or `hidden off` run without requiring code changes.",
 keywords: &["hidden", "hidden line removal", "hidden on", "hidden off", "matlab hidden"],
 related: &["mesh", "surf", "shading", "get", "set"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`hidden on` sets `HiddenLineRemoval` to `'on'` on the current axes.",
"`hidden off` sets `HiddenLineRemoval` to `'off'` on the current axes.",
"`hidden` with no mode toggles the current axes state.",
"`hidden(ax, mode)` and `hidden(ax)` operate on the supplied axes handle without changing unrelated axes.",
"When hidden-line removal is off, mesh edges and wireframe surface lines use a no-depth line pipeline so obscured lines remain visible. Filled surface faces continue to render through the normal surface pipeline.",
"An assigned logical state/success result is a RunMat extension. MATLAB compatibility mode allows command calls but rejects an explicitly requested result before changing state."
] },
BuiltinDocumentationSection { heading: "Placement and display", paragraphs: &[
"`hidden` is a graphics-state command. It does not inspect or gather numeric plot data, and it does not add provider kernels. The no-depth line behavior is implemented in `runmat-plot`'s renderer for mesh and wireframe surface render data, so existing GPU-resident plot buffers remain in their plot renderer paths.",
"These commands update shared graphics state and may notify figure observers. Numeric array acceleration remains with the original plotting/rendering pipeline; the commands do not launch array-provider kernels. Graphics display requires a browser graphics host or a desktop graphics host.",
"Control arguments must already be host values. Unsupported resident controls reject without provider downloads; existing resident plot buffers stay owned by their plotting/rendering paths. Invalid syntax and excess output requests are checked before creating or changing graphics state."
] }
 ],
 examples: &[
BuiltinExample {
 id: "example-1",
 title: "Disable hidden-line removal for a mesh workflow",
 program: "[X, Y, Z] = peaks(30);\nmesh(X, Y, Z);\nhidden off;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(gca,'HiddenLineRemoval'),'off'));" },
},
BuiltinExample {
 id: "example-2",
 title: "Toggle a specific subplot",
 program: "ax1 = subplot(1, 2, 1);\nsurf(peaks(20));\nax2 = subplot(1, 2, 2);\nmesh(peaks(20));\nhidden(ax1, 'off');\nhidden(ax2, 'on');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(ax1,'HiddenLineRemoval'),'off')); assert(strcmp(get(ax2,'HiddenLineRemoval'),'on'));" },
},
BuiltinExample {
 id: "example-3",
 title: "Query or set the axes property directly",
 program: "plot([1 2],[2 1]); hidden off; state = get(gca,'HiddenLineRemoval'); set(gca,'HiddenLineRemoval','on');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(state,'off')); assert(strcmp(get(gca,'HiddenLineRemoval'),'on'));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Is hidden the same as making a plot invisible?", answer: "No. `hidden` controls hidden-line-removal state for surface-family rendering. Use the `Visible` property to show or hide graphics objects." },
BuiltinDocumentationFaq { question: "Can I use normal property syntax instead?", answer: "Yes. The command updates the same axes property that `get(gca, 'HiddenLineRemoval')` and `set(gca, 'HiddenLineRemoval', 'off')` use." },
BuiltinDocumentationFaq { question: "Does hidden affect every subplot?", answer: "No. Hidden-line-removal state is axes-local. Use `hidden(ax, mode)` when you want to update a specific axes." }
 ],
 links: &[
BuiltinDocumentationLink { label: "mesh", target: BuiltinDocumentationLinkTarget::Builtin("mesh") },
BuiltinDocumentationLink { label: "surf", target: BuiltinDocumentationLinkTarget::Builtin("surf") },
BuiltinDocumentationLink { label: "shading", target: BuiltinDocumentationLinkTarget::Builtin("shading") },
BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/hidden.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "hidden state, output and validation tests", location: "builtins::plotting::cmds::hidden::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
