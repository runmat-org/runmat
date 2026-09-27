use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("cla"),
 slug: Some("cla"),
 summary: "Clear the current axes in MATLAB and RunMat plotting.",
 description: "`cla` clears the current or specified axes without clearing other axes or selecting a different target. `cla('reset')` and `cla(ax,'reset')` also reset axes appearance while preserving axes identity, position, units, and independently owned colorbars.",
 keywords: &["cla", "clear axes", "plotting", "subplot"],
 related: &["clf", "subplot", "gca", "hold"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`cla()` clears the active axes.",
"The current subplot selection determines which axes are cleared.",
"Other axes in the same figure are left untouched.",
"`cla` resets the line-style cycle for the cleared axes so subsequent plots start from the default style sequence again.",
"`cla(ax)` targets an existing axes without changing the current axes. Cleared plot handles become invalid; surviving handles in other axes continue to refer to the same plots.",
"Reset restores defaults including the figure's inherited colormap, but keeps the axes Position and Units. The assigned logical success output is a RunMat extension; MATLAB compatibility mode accepts command calls only."
] },
BuiltinDocumentationSection { heading: "Placement and display", paragraphs: &[
"`cla` clears plotting state only. It does not execute numeric GPU kernels.",
"`cla` only mutates plotting state and does not affect array residency.",
"`cla` performs no provider dispatch; it clears the active axes from the plotting registry and renderer state.",
"These commands update shared graphics state and may notify figure observers. Numeric array acceleration remains with the original plotting/rendering pipeline; the commands do not launch array-provider kernels. Graphics display requires a browser graphics host or a desktop graphics host.",
"Control arguments must already be host values. Unsupported resident controls reject without provider downloads; existing resident plot buffers stay owned by their plotting/rendering paths. Invalid syntax and excess output requests are checked before creating or changing graphics state."
] }
 ],
 examples: &[
BuiltinExample {
 id: "example-1",
 title: "Clear only the active subplot",
 program: "ax1 = subplot(1,2,1); h1 = plot(1:5,1:5);\nax2 = subplot(1,2,2); h2 = plot(1:5,[5 4 3 2 1]); cla();",
 display_output: Some("% only the second subplot is cleared"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h1)); assert(~isgraphics(h2)); assert(gca == ax2); assert(isequal(get(h1,'YData'),1:5));" },
},
BuiltinExample {
 id: "example-2",
 title: "Clear the current axes and plot again",
 program: "old = plot(1:5,1:5); cla(); h = plot(1:5,[1 2 3 4 5]);",
 display_output: Some("% the second plot starts on an empty axes"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(~isgraphics(old)); assert(isequal(get(h,'YData'),[1 2 3 4 5]));" },
},
BuiltinExample {
 id: "example-3",
 title: "Reset an explicit target without selecting it",
 program: "ax1 = subplot(1,2,1); h1 = plot([1 2],[2 3]); axis([0 10 0 10]);\nax2 = subplot(1,2,2); h2 = plot([1 2],[3 4]); cla(ax1,'reset');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(gca == ax2); assert(~isgraphics(h1)); assert(isgraphics(h2)); assert(isequal(axis(ax1),[0 1 0 1]));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "How is `cla` different from `clf`?", answer: "`cla` clears only the current axes, while `clf` clears the entire current figure." },
BuiltinDocumentationFaq { question: "Does `cla` affect other subplots?", answer: "No. Only the current axes, or the explicitly supplied axes, is cleared. Other subplots and their plot handles remain intact." },
BuiltinDocumentationFaq { question: "Does `cla` run on the GPU?", answer: "No. It only mutates host-side plotting state." }
 ],
 links: &[
BuiltinDocumentationLink { label: "clf", target: BuiltinDocumentationLinkTarget::Builtin("clf") },
BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
BuiltinDocumentationLink { label: "gca", target: BuiltinDocumentationLinkTarget::Builtin("gca") },
BuiltinDocumentationLink { label: "hold", target: BuiltinDocumentationLinkTarget::Builtin("hold") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/cla.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "cla state, output and validation tests", location: "builtins::plotting::cmds::cla::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
