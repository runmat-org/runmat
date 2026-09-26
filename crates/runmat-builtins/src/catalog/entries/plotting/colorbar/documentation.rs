use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("colorbar"),
 slug: Some("colorbar"),
 summary: "Show, hide, or toggle colorbars in MATLAB and RunMat plotting.",
 description: "`colorbar` controls the color scale associated with the current or specified axes. Each enabled colorbar has a registered graphics identity and an independent lifetime. Default RunMat behavior preserves the historical toggle and logical result; MATLAB compatibility mode enables and returns a ColorBar handle.",
 keywords: &["colorbar", "color scale", "image colorbar", "surface colorbar", "matlab colorbar"],
 related: &["colormap", "imagesc", "surf", "contourf"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`colorbar on` enables the colorbar for the current axes.",
"`colorbar off` disables the colorbar for the current axes.",
"With no mode, RunMat toggles and returns logical enabled state. MATLAB compatibility mode enables an existing or new colorbar and returns its registered handle; repeated calls reuse that handle.",
"Colorbar visibility is subplot-local and does not automatically affect other axes.",
"An optional leading axes handle targets that axes without changing selection. `off` deletes the associated ColorBar identity. Setting a ColorBar handle's Visible property to false hides the live object without deleting its identity.",
"ColorBar handles expose Type, Parent, Children, and Visible. Position, Location, Labels, and colorbar-specific label children are not yet supported. MATLAB-profile `off` calls have no output."
] },
BuiltinDocumentationSection { heading: "Placement and display", paragraphs: &[
"These commands update shared graphics state and may notify figure observers. Numeric array acceleration remains with the original plotting/rendering pipeline; the commands do not launch array-provider kernels. Graphics display requires a browser graphics host or a desktop graphics host.",
"Control arguments must already be host values. Unsupported resident controls reject without provider downloads; existing resident plot buffers stay owned by their plotting/rendering paths. Invalid syntax and excess output requests are checked before creating or changing graphics state."
] }
 ],
 examples: &[
BuiltinExample {
 id: "example-1",
 title: "Show a colorbar for a scaled image",
 program: "[X, Y] = meshgrid(linspace(-3, 3, 60), linspace(-3, 3, 60));\nA = sin(X) .* cos(Y);\nimagesc(A);\ncolorbar on;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(get(gca,'Colorbar'));" },
},
BuiltinExample {
 id: "example-2",
 title: "Use colorbars independently across subplots",
 program: "[X,Y] = meshgrid(linspace(-3,3,40)); Z = sin(X).*cos(Y);\nax1 = subplot(1,2,1); contourf(X,Y,Z); colorbar on;\nax2 = subplot(1,2,2); surf(X,Y,Z); colorbar off;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(get(ax1,'Colorbar')); assert(~get(ax2,'Colorbar'));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "How do I add a label to the colorbar?", answer: "ColorBar-specific label children and `ylabel(colorbarHandle,...)` are not yet supported. You can annotate the axes with `ylabel('Intensity')`, but that labels the axes, not the colorbar. For example, `imagesc(peaks(50)); colorbar on; ylabel('Intensity');` adds an axes label alongside the scale." },
BuiltinDocumentationFaq { question: "Can I limit the colorbar range without clipping the data?", answer: "Set `caxis([cmin cmax])` (or `clim([cmin cmax])`) on the axes to control the color mapping range. Values outside that range get clamped to the endpoint colors, but the underlying data stays intact." },
BuiltinDocumentationFaq { question: "Does the colorbar track across subplots automatically?", answer: "No. Colorbar state is subplot-local, so each axes manages its own colorbar independently. You need to call `colorbar on` in each subplot where you want one visible." }
 ],
 links: &[
BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
BuiltinDocumentationLink { label: "imagesc", target: BuiltinDocumentationLinkTarget::Builtin("imagesc") },
BuiltinDocumentationLink { label: "surf", target: BuiltinDocumentationLinkTarget::Builtin("surf") },
BuiltinDocumentationLink { label: "contourf", target: BuiltinDocumentationLinkTarget::Builtin("contourf") },
BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/colorbar.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "colorbar state, output and validation tests", location: "builtins::plotting::cmds::colorbar::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
