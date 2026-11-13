use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The returned value is a text handle that can be inspected or updated through `get` and `set`.",
            "With no explicit target, `title` attaches text to the current axes as returned by `gca`.",
            "Passing an axes handle as the first argument targets that specific axes, which is useful in subplot layouts.",
            "String scalars, char arrays, string arrays, cell arrays, and finite numeric scalars are accepted. Native integer scalars are formatted directly from their exact decimal value, including full-width `int64` and `uint64` values. String arrays and cell arrays produce multiline titles.",
            "Text styling properties — `FontSize`, `FontWeight`, `FontAngle`, `Color`, `Interpreter`, and `Visible` — use the shared plotting text-property system.",
            "`FontSize` accepts single, double, and all eight native integer classes, then validates that the size is positive, finite, and within the renderer's supported range.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and execution",
        paragraphs: &[
            "This command mutates local graphics scene state and may notify the graphics host. Inputs and returned graphics handles are host-owned; worker transfer of numeric handle values does not transfer scene ownership.",
            "Interpreter and compiled routes use the same runtime operation; a browser graphics host or desktop graphics host is required for presentation.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Example captions",
        paragraphs: &[
            "% title is rendered above the axes",
            "% Each panel has its own title; the figure has none",
            "% Titles are attached to their respective axes regardless of which is currently selected",
            "% Both strings appear, stacked above the axes",
            "% The axes title is updated in place",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Add a title to the current axes",
        program: "plot(0:0.1:2*pi, sin(0:0.1:2*pi));\ntitle('Sine Wave');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'Title'),'String'),'Sine Wave'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Style the title and inspect its handle",
        program: "plot(1:10, rand(1, 10));\nh = title('Random Signal', 'FontSize', 16, 'FontWeight', 'bold');\nget(h, 'Type')",
        display_output: Some("ans =\n    'text'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'Random Signal')); assert(get(h,'FontSize')==16);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Set an independent title on each subplot panel",
        program: "subplot(1, 2, 1);\nplot(0:0.1:pi, sin(0:0.1:pi));\ntitle('Sine');\nsubplot(1, 2, 2);\nplot(0:0.1:pi, cos(0:0.1:pi));\ntitle('Cosine');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'Title'),'String'),'Cosine'));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Target a specific axes using an explicit handle",
        program: "ax1 = subplot(1, 2, 1);\nplot(ax1, 1:5, 1:5);\nax2 = subplot(1, 2, 2);\nplot(ax2, 1:5, (1:5).^2);\ntitle(ax1, 'Linear');\ntitle(ax2, 'Quadratic');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(ax1,'Title'),'String'),'Linear')); assert(strcmp(get(get(ax2,'Title'),'String'),'Quadratic'));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Create a multiline title from a string array",
        program: "plot(1:5, [2 4 1 3 5]);\ntitle([\"Top line\"; \"Bottom line\"]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(numel(get(get(gca,'Title'),'String'))==2);",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Update a title after plotting using set",
        program: "plot(1:10, rand(1, 10));\nh = title('Draft');\nset(h, 'String', 'Final Title', 'FontSize', 14);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'Final Title')); assert(get(h,'FontSize')==14);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("title"),
    slug: Some("title"),
    summary: "Set the title of the current axes or specified axes.",
    description: "`title` attaches text to individual axes and returns a text handle for later property updates. Use `sgtitle` when one heading should span an entire figure.",
    keywords: &["title", "axes title", "plot title", "label", "matlab title"],
    related: &["sgtitle", "xlabel", "ylabel", "subplot", "get", "set"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `title` different from `sgtitle`?",
            answer: "`title` is axes-scoped: it attaches to one panel. `sgtitle` is figure-scoped and is drawn once above all subplot panels. Use `title` for per-panel labels and `sgtitle` for an overall figure heading.",
        },
        BuiltinDocumentationFaq {
            question: "How do I create a multiline title?",
            answer: "Pass a string array or cell array. RunMat joins the elements with newlines.\n\n```matlab\ntitle({'First line', 'Second line'});\n```",
        },
        BuiltinDocumentationFaq {
            question: "Can I update the title after the plot is created?",
            answer: "Yes. `title` returns a text handle, so you can update it at any time with `set`.\n\n```matlab\nh = title('Initial');\nset(h, 'String', 'Updated', 'Color', 'r');\n```",
        },
        BuiltinDocumentationFaq {
            question: "How do I set titles on specific subplots without changing the active axes?",
            answer: "Pass the axes handle returned by `subplot` or `gca` as the first argument.\n\n```matlab\nax = subplot(2, 1, 1);\ntitle(ax, 'Top Panel');\n```",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "sgtitle",
            target: BuiltinDocumentationLinkTarget::Builtin("sgtitle"),
        },
        BuiltinDocumentationLink {
            label: "xlabel",
            target: BuiltinDocumentationLinkTarget::Builtin("xlabel"),
        },
        BuiltinDocumentationLink {
            label: "ylabel",
            target: BuiltinDocumentationLinkTarget::Builtin("ylabel"),
        },
        BuiltinDocumentationLink {
            label: "subplot",
            target: BuiltinDocumentationLinkTarget::Builtin("subplot"),
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
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/title.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/title/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/title.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
