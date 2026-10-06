use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The returned value is a text handle that can be inspected or updated through `get` and `set`.",
            "Unlike `title`, `sgtitle` is figure-scoped and is drawn once above all subplot panels.",
            "String scalars, char arrays, string arrays, and cell arrays can all be used to create multiline super titles.",
            "Text styling properties such as `FontSize`, `FontWeight`, `FontAngle`, `Color`, `Interpreter`, and `Visible` use the shared plotting text-property system.",
            "A scalar value from any integer class can be used as title text; RunMat formats the exact signed or unsigned value directly, including `uint64` values above the exact double range. Integer `FontSize` values are accepted as numeric style controls.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and execution",
        paragraphs: &[
            "This command mutates local graphics scene state and may notify the graphics host. Inputs and returned graphics handles are host-owned; worker transfer of numeric handle values does not transfer scene ownership.",
            "Interpreter and compiled routes use the same runtime operation; a browser graphics host or desktop graphics host is required for presentation.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Add a super title above a subplot figure",
        program: "subplot(2, 2, 1);\nplot(1:10, rand(1, 10));\nsubplot(2, 2, 2);\nplot(1:10, rand(1, 10));\nsubplot(2, 2, 3);\nplot(1:10, rand(1, 10));\nsubplot(2, 2, 4);\nplot(1:10, rand(1, 10));\nsgtitle('Experiment Summary');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gcf,'SGTitle'),'String'),'Experiment Summary'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Style the figure-level title and inspect the returned handle",
        program: "subplot(1, 2, 1);\nimagesc(peaks(20));\nsubplot(1, 2, 2);\nimagesc(peaks(20));\nh = sgtitle('Field Comparison', 'FontSize', 18, 'FontWeight', 'bold');\nget(h, 'Type')",
        display_output: Some("ans =\n    'text'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'Field Comparison')); assert(strcmp(get(h,'Type'),'text'));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Target an explicit figure handle",
        program: "fig = figure(42);\nsubplot(1, 2, 1);\nplot(0:0.1:1, sin(0:0.1:1));\nsubplot(1, 2, 2);\nplot(0:0.1:1, cos(0:0.1:1));\nsgtitle(fig, 'Sine and Cosine');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(fig,'SGTitle'),'String'),'Sine and Cosine'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sgtitle"),
    slug: Some("sgtitle"),
    summary: "Set a centered figure-level title above all subplots with MATLAB-compatible `sgtitle` behavior.",
    description: "`sgtitle` sets or updates a single figure-level title rendered above the full figure instead of a specific axes. It supports MATLAB-compatible current-figure and explicit-figure forms and returns a text handle for property updates.",
    keywords: &["sgtitle", "super title", "subplot title", "figure title", "matlab sgtitle"],
    related: &["title", "subplot", "figure", "get", "set"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is sgtitle different from title?",
            answer: "`title` attaches text to the current axes, so each subplot can have its own panel title. `sgtitle` attaches text to the figure as a whole and is drawn once above all panels. Use `title` for per-panel labels and `sgtitle` for the overall figure heading.",
        },
        BuiltinDocumentationFaq {
            question: "Can I update the super title after creating it?",
            answer: "Yes. `sgtitle` returns a text handle, so you can inspect or update it with `get` and `set`.\n\n```matlab\nh = sgtitle('Initial Title');\nset(h, 'String', 'Updated Title', 'FontWeight', 'bold');\n```",
        },
        BuiltinDocumentationFaq {
            question: "What target objects are supported?",
            answer: "Phase 1 supports the current figure by default and explicit figure handles through `sgtitle(fig, txt, ...)`. Larger MATLAB target containers such as panels or tabs are tracked separately and are not part of the current implementation.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "title",
            target: BuiltinDocumentationLinkTarget::Builtin("title"),
        },
        BuiltinDocumentationLink {
            label: "subplot",
            target: BuiltinDocumentationLinkTarget::Builtin("subplot"),
        },
        BuiltinDocumentationLink {
            label: "figure",
            target: BuiltinDocumentationLinkTarget::Builtin("figure"),
        },
        BuiltinDocumentationLink {
            label: "get",
            target: BuiltinDocumentationLinkTarget::Builtin("get"),
        },
        BuiltinDocumentationLink {
            label: "set",
            target: BuiltinDocumentationLinkTarget::Builtin("set"),
        },
        BuiltinDocumentationLink {
            label: "Styling plots and axes",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes"),
        },
        BuiltinDocumentationLink {
            label: "Plot replay and export",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export"),
        },
        BuiltinDocumentationLink {
            label: "Complete plotting guide",
            target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/sgtitle.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/sgtitle/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/sgtitle.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
