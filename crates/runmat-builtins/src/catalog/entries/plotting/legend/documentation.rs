use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Calling `legend` with no label arguments uses plotted object display names when they are available.",
            "Labels can be passed as separate strings, string arrays, or cell arrays, following common MATLAB `legend` workflows.",
            "Legend visibility and style are axes-local, so enabling or hiding a legend in one subplot does not mutate every axes in the figure.",
            "The returned handle can be used with `get` and `set` to inspect or update location, orientation, box visibility, font styling, and related legend properties.",
            "For a subplot-local legend, call `legend` on that axes. A legend snapshot does not automatically gain entries for series added later; call `legend` again after adding them.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Legend commands operate on host graphics state. They do not run a numeric accelerator kernel or gather resident labels or style values.",
            "A numeric graphics handle identifies local figure state, not a transferable graphics object. Invalid axes, labels, and style properties are reported by the graphics parser or state owner.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Build a legend from display names",
        program: "x = linspace(0, 2*pi, 200);\nh1 = plot(x, sin(x));\nset(h1, 'DisplayName', 'sin(x)');\nhold on;\nh2 = plot(x, cos(x));\nset(h2, 'DisplayName', 'cos(x)');\nlegend;",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "lg = legend; assert(strcmp(get(lg,'Type'),'legend'));",
        },
    },
    BuiltinExample {
        id: "example-2", title: "Pass labels explicitly and position the legend",
        program: "plot(1:5, [1 2 3 4 5]);\nhold on;\nplot(1:5, [5 4 3 2 1]);\nlg = legend('rising', 'falling', 'Location', 'southwest');\nget(lg, 'Type')",
        display_output: Some("ans =\n    'legend'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(lg,'Type'),'legend')); assert(strcmp(get(lg,'Location'),'southwest'));",
        },
    },
    BuiltinExample {
        id: "example-3", title: "Keep legends subplot-local",
        program: "subplot(1, 2, 1);\nplot(1:5, 1:5);\nlg = legend('left');\nsubplot(1, 2, 2);\nplot(1:5, [5 4 3 2 1]);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(lg,'Type'),'legend'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("legend"), slug: Some("legend"),
    summary: "Show and configure plot legends in MATLAB and RunMat.",
    description: "`legend` creates or updates the legend for the current axes (or an explicit axes handle). Returned handle behavior, display-name sourcing, and style controls follow MATLAB semantics.",
    keywords: &["legend", "chart legend", "display names", "matlab legend", "series labels"],
    related: &["subplot", "get", "set", "plot"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How do I move the legend to a specific corner?",
            answer: "Pass the `'Location'` property when creating the legend, or update it afterward with `set`. Standard MATLAB location strings work: `'northeast'`, `'southwest'`, `'northwest'`, `'southeast'`, etc.\n\n```matlab\nlg = legend('a', 'b');\nset(lg, 'Location', 'northwest');\n```",
        },
        BuiltinDocumentationFaq {
            question: "Can I show only some series in the legend?",
            answer: "Set `'DisplayName'` only on the series you want labeled, then call `legend` with no arguments. Series without a display name are excluded. Alternatively, pass explicit label strings to `legend` — only the first N plotted objects get entries.",
        },
        BuiltinDocumentationFaq {
            question: "Do I need to call legend again after adding more series with hold on?",
            answer: "Yes. The legend snapshot is taken when `legend` is called, so new series added after the initial `legend` call won't appear until you call `legend` again. Set display names on all series first, then call `legend` once at the end.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
        BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/legend.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Legend runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/legend.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Legend runtime tests", location: "builtins::plotting::legend::tests" }],
        notes: &["The legacy subplot example's output was a descriptive comment, not program output; its meaning is retained in the Behavior section and the figure oracle. Browser/Desktop presentation remains a separate execution gate."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
