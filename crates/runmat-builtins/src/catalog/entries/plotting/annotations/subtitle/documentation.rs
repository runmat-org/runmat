use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The returned value is a text handle that can be inspected or updated through `get` and `set`.",
            "With no explicit target, `subtitle` attaches text to the current axes as returned by `gca`.",
            "Passing an axes handle as the first argument targets that specific axes, which is useful in subplot layouts.",
            "String scalars, char arrays, string arrays, and cell arrays are all accepted. String arrays and cell arrays produce multiline subtitles, with elements joined by newlines.",
            "Text styling properties such as `FontSize`, `FontWeight`, `FontAngle`, `Color`, `Interpreter`, and `Visible` use the shared plotting text-property system.",
            "`FontSize` accepts every native integer class as well as floating-point scalars. The value crosses the graphics floating boundary only after exact scalar validation.",
            "Axes handles are normally the double handles returned by plotting functions. Typed-integer aliases for encoded axes handles are available only when RunMat extensions are enabled and are rejected in strict compatibility mode before graphics state changes.",
            "`get(ax, 'Subtitle')` returns the subtitle text handle for that axes. `get(h, 'String')` returns the subtitle text for the subtitle handle.",
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
            "% title and subtitle are rendered above the axes",
            "% Subtitles are attached to their respective axes",
            "% Both cell elements appear as separate subtitle lines",
            "% The axes subtitle is updated in place",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Add a title and subtitle to the current axes",
        program: "plot(0:0.1:2*pi, sin(0:0.1:2*pi));\ntitle('Sine Wave');\nsubtitle('One period sampled every 0.1 radians');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'Subtitle'),'String'),'One period sampled every 0.1 radians'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Style the subtitle and inspect its handle",
        program: "plot(1:10, rand(1, 10));\nh = subtitle('Trial A', 'FontSize', 12, 'FontWeight', 'bold');\nget(h, 'Type')",
        display_output: Some("ans =\n    'text'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'Trial A')); assert(strcmp(get(h,'Type'),'text'));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Set independent subtitles on subplot panels",
        program: "ax1 = subplot(1, 2, 1);\nplot(ax1, 1:5, 1:5);\nax2 = subplot(1, 2, 2);\nplot(ax2, 1:5, (1:5).^2);\nsubtitle(ax1, 'Linear response');\nsubtitle(ax2, 'Quadratic response');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(ax1,'Subtitle'),'String'),'Linear response')); assert(strcmp(get(get(ax2,'Subtitle'),'String'),'Quadratic response'));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Create a multiline subtitle from a cell array",
        program: "plot(1:5, [2 4 1 3 5]);\nsubtitle({'Experiment 7', 'Filtered signal'});",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(numel(get(get(gca,'Subtitle'),'String'))==2);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Update a subtitle after plotting using set",
        program: "plot(1:10, rand(1, 10));\nh = subtitle('Draft');\nset(h, 'String', 'Final Subtitle', 'Color', 'r');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'Final Subtitle'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("subtitle"),
    slug: Some("subtitle"),
    summary: "Set subtitle text for the current axes or a specified axes handle with MATLAB-compatible `subtitle` forms.",
    description: "`subtitle` attaches secondary title text to individual axes and returns a text handle for property updates. It follows MATLAB-compatible axes-scoped behavior and is separate from figure-level title helpers such as `sgtitle` and `suptitle`.",
    keywords: &["subtitle", "axes subtitle", "plot subtitle", "label", "matlab subtitle"],
    related: &["title", "sgtitle", "subplot", "get", "set"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `subtitle` different from `title`?",
            answer: "`title` sets the primary axes title. `subtitle` sets a second text line associated with the same axes and is rendered below the primary title when both are present.",
        },
        BuiltinDocumentationFaq {
            question: "Can I target a specific subplot?",
            answer: "Yes. Pass the axes handle returned by `subplot` or `gca` as the first argument.\n\n```matlab\nax = subplot(2, 1, 1);\nsubtitle(ax, 'Top panel details');\n```",
        },
        BuiltinDocumentationFaq {
            question: "Can I update the subtitle after creating it?",
            answer: "Yes. `subtitle` returns a text handle, so you can update it with `set`.\n\n```matlab\nh = subtitle('Initial');\nset(h, 'String', 'Updated', 'FontSize', 12);\n```",
        },
        BuiltinDocumentationFaq {
            question: "Can `FontSize` be an integer value?",
            answer: "Yes. All eight native integer classes are accepted for `FontSize`, and practical positive sizes are validated exactly before entering the renderer's floating-point graphics domain.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "title",
            target: BuiltinDocumentationLinkTarget::Builtin("title"),
        },
        BuiltinDocumentationLink {
            label: "sgtitle",
            target: BuiltinDocumentationLinkTarget::Builtin("sgtitle"),
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
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/subtitle.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/subtitle/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/subtitle.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
