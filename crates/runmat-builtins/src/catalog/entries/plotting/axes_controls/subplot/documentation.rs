use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("subplot"),
    slug: Some("subplot"),
    summary: "Select subplot grid locations to create multi-panel axes layouts with MATLAB-compatible `subplot` behavior.",
    description: "`subplot(m, n, p)` selects or creates an axes position inside an `m`-by-`n` figure grid and returns its handle. It follows MATLAB-compatible subplot indexing and current-axes selection behavior.",
    keywords: &["subplot", "subplot grid", "multi-panel chart", "matlab subplot", "figure layout"],
    related: &["plot", "legend", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "The returned value is an axes handle that can be passed explicitly to later plotting commands.",
            "Subplot-local metadata stays isolated per axes: titles, legends, limits, log modes, and 3-D view state are all tracked independently.",
            "Subsequent plotting commands operate on the selected subplot unless another axes handle is passed explicitly.",
            "Indices are one-based in the MATLAB style. `subplot(2, 2, 3)` selects the third slot in row-major order.",
            "The documented grid controls use single or double values. RunMat can additionally parse typed-integer scalar controls exactly when RunMat extensions are enabled; strict compatibility mode rejects that extra form before changing figure state.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics state remains host-owned. Resident control values reject before provider access; resident plotted arrays retain the rendering pipeline.",
            "Whole-call resource accounting, cooperative large host controls, and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Create a 2x2 subplot layout", program: "subplot(2, 2, 1);\nplot(0:0.1:1, (0:0.1:1).^2);\nsubplot(2, 2, 2);\nscatter(1:5, [5 4 3 2 1]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isnumeric(gca()));" } },
        BuiltinExample { id: "example-2", title: "Capture and reuse an axes handle explicitly", program: "ax = subplot(1, 2, 2);\nplot(ax, 0:0.1:1, sin(0:0.1:1));\nget(ax, 'Type')", display_output: Some("ans =\n    'axes'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(ax,'Type'),'axes'));" } },
        BuiltinExample { id: "example-3", title: "Show that subplot-local state stays isolated caption: % Only the first subplot uses a logarithmic x-axis", program: "subplot(1, 2, 1);\nsemilogx(logspace(0, 2, 50), logspace(0, 2, 50));\nsubplot(1, 2, 2);\nplot(1:5, 1:5);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(subplot(1,2,1),'XScale'),'log'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How do I control the number of rows and columns in a subplot layout?", answer: "The first two arguments to `subplot(m, n, p)` set the grid dimensions: `m` rows and `n` columns. The third argument `p` selects which slot to activate, counted in row-major order starting at 1. `subplot(3, 1, 2)` gives you the middle row of a three-row single-column layout." },
        BuiltinDocumentationFaq { question: "Can two subplots share the same x-axis range?", answer: "Set matching limits on both axes manually with `axis` or `set`. RunMat also provides `linkaxes` for supported linked-axis groups; explicit matching limits remain available as illustrated below.\n\n```matlab\nsubplot(2, 1, 1); plot(0:10, rand(1, 11)); axis([0 10 0 1]);\nsubplot(2, 1, 2); plot(0:10, rand(1, 11)); axis([0 10 0 1]);\n```" },
        BuiltinDocumentationFaq { question: "Can I make subplots of unequal sizes?", answer: "No. This implementation requires a scalar subplot position; vector positions that span multiple grid slots are not supported. Use separate scalar positions to create the individual panels. A vector position is rejected before configuring the subplot." },
        BuiltinDocumentationFaq { question: "Can the grid controls be typed integers?", answer: "The compatibility surface documents single and double controls. If RunMat extensions are enabled, all eight native integer scalar classes are also accepted and parsed exactly; strict compatibility mode rejects that extra form." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "legend", target: BuiltinDocumentationLinkTarget::Builtin("legend") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
        BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/subplot.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "subplot behavior and validation tests", location: "builtins::plotting::subplot::tests" }],
        notes: &["Legacy behaviors, examples, explanatory output captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
