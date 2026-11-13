use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("shading"),
 slug: Some("shading"),
 summary: "Set surface shading to flat, interpolated, or faceted modes with MATLAB-compatible `shading` syntax.",
 description: "`shading` controls how surface-family plots render color interpolation and facet edges. It supports MATLAB-compatible forms such as `shading flat`, `shading interp`, and `shading faceted` for `surf`/`mesh` workflows.",
 keywords: &["shading", "surface shading", "shading interp", "shading flat", "matlab shading"],
 related: &["surf", "mesh", "colormap", "colorbar"],
 sections: &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"`shading flat` uses flat shading across faces.",
"`shading interp` enables interpolated/smooth shading across the rendered surface.",
"`shading faceted` restores faceted surface presentation.",
"The command changes existing surface-family objects on the current axes only; it does not set a default for future surfaces or modify unrelated subplots.",
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
 title: "Apply interpolated shading to a surface",
 program: "[X,Y] = meshgrid(linspace(-3,3,50)); Z = sin(X).*cos(Y); h = surf(X,Y,Z); shading interp;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Shading'),'interp'));" },
},
BuiltinExample {
 id: "example-2",
 title: "Compare flat and faceted shading across subplots",
 program: "[X,Y] = meshgrid(linspace(-3,3,40)); Z = sin(X).*cos(Y);\nsubplot(1,2,1); h1 = surf(X,Y,Z); shading flat;\nsubplot(1,2,2); h2 = surf(X,Y,Z); shading faceted;",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h1,'Shading'),'flat')); assert(strcmp(get(h2,'Shading'),'faceted'));" },
}
],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "What's the difference between interp, flat, and faceted shading?", answer: "`flat` fills each face with a single color (the value at one vertex). `interp` smoothly interpolates color across each face, producing a continuous gradient. `faceted` is like `flat` but with visible black edge lines between faces — this is the default `surf` appearance." },
BuiltinDocumentationFaq { question: "When should I use shading interp vs faceted?", answer: "Use `interp` when you want a smooth, publication-quality surface without visible grid edges — good for dense meshes or photorealistic renders. Use `faceted` (the default) when you want to see the mesh structure, which helps during debugging or when the grid resolution matters to the reader." },
BuiltinDocumentationFaq { question: "Does shading affect mesh plots or just surf?", answer: "It updates existing surface-backed objects created by surf and mesh, including the surface component of supported composite plots such as surfc and meshc. It does not change independent patch objects or set a default for future objects. Edge visibility also depends on the surface's wireframe and edge settings." }
 ],
 links: &[
BuiltinDocumentationLink { label: "surf", target: BuiltinDocumentationLinkTarget::Builtin("surf") },
BuiltinDocumentationLink { label: "mesh", target: BuiltinDocumentationLinkTarget::Builtin("mesh") },
BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
BuiltinDocumentationLink { label: "colorbar", target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },
BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cmds/shading.rs") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "shading state, output and validation tests", location: "builtins::plotting::cmds::shading::tests" }],
 notes: &[
 "Interpreter, JIT and AOT use the canonical runtime binding. Runtime state tests and source checks do not prove presentation; BrowserGraphics Figure examples require actual browser surfaces and presentation, and desktop engine witnesses require a graphics host.",
 "Legacy documentation topics, FAQ guidance, related links and plotting examples are preserved with corrections for implemented shared state and explicitly bounded compatibility. The renderer integration suite is runmat-plot/tests/renderer_tests.rs.",
 ],
 },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),
};
