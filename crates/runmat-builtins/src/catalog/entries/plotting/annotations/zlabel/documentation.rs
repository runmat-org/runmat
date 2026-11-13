use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The returned value is a text handle that can be queried and updated through `get` and `set`.",
            "String scalars, string arrays, cell arrays, and numeric scalars can be used as label text. Native integer scalars use their exact decimal spelling.",
            "`FontSize` accepts single, double, and all eight native integer classes; values must be positive and finite.",
            "Z-axis labels are subplot-local and stay attached to the axes they were created for. RunMat mode additionally accepts typed-integer aliases for encoded axes handles.",
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
        title: "Add a z-axis label to a surface plot",
        program: "[X, Y] = meshgrid(linspace(-3, 3, 50), linspace(-3, 3, 50));\nZ = sin(X) .* cos(Y);\nsurf(X, Y, Z);\nzlabel('Height');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'ZLabel'),'String'),'Height'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Create a multiline z-axis label and style it",
        program: "[X, Y] = meshgrid(-2:0.2:2, -2:0.2:2);\nZ = X .* exp(-X.^2 - Y.^2);\nsurf(X, Y, Z);\nh = zlabel([\"Depth\", \"(km)\"]);\nset(h, 'FontWeight', 'bold');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(numel(get(h,'String'))==2);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Target a specific subplot axes",
        program: "[X, Y] = meshgrid(linspace(-3, 3, 30), linspace(-3, 3, 30));\nZ = sin(X) .* cos(Y);\nsubplot(1, 2, 1);\nsurf(X, Y, Z);\nax = subplot(1, 2, 2);\nplot3(cos(0:0.1:5), sin(0:0.1:5), 0:0.1:5);\nzlabel(ax, 'Trajectory Height');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(ax,'ZLabel'),'String'),'Trajectory Height'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("zlabel"),
    slug: Some("zlabel"),
    summary: "Set z-axis label text for current or specified axes with MATLAB-compatible `zlabel` forms.",
    description: "`zlabel` sets or updates z-axis label text and returns a text handle for styling/property edits, following MATLAB-compatible axes-label workflows.",
    keywords: &["zlabel", "z-axis label", "3-D axis label", "matlab zlabel", "surface axis labeling"],
    related: &["view", "plot3", "surf", "get", "set"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "When do I need zlabel vs just xlabel/ylabel?",
            answer: "`zlabel` only applies to 3-D axes — `surf`, `mesh`, `plot3`, `scatter3`, etc. For 2-D plots, the z-axis doesn't exist, so `zlabel` has no effect. If you're plotting in 3-D, always label all three axes so readers can orient themselves.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use LaTeX formatting in zlabel?",
            answer: "You can use LaTeX-style inline math in the label string with standard TeX markup. Subscripts, superscripts, and Greek letters work in label text.\n\n```matlab\nzlabel('Temperature (\\circC)');\nzlabel('\\sigma_{z} (MPa)');\n```",
        },
        BuiltinDocumentationFaq {
            question: "How do I style the z-axis label after creating it?",
            answer: "`zlabel` returns a text handle that works with `get` and `set`. You can change font weight, size, color, and other text properties through that handle.\n\n```matlab\nh = zlabel('Elevation (m)');\nset(h, 'FontWeight', 'bold', 'FontSize', 14);\n```",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "view",
            target: BuiltinDocumentationLinkTarget::Builtin("view"),
        },
        BuiltinDocumentationLink {
            label: "plot3",
            target: BuiltinDocumentationLinkTarget::Builtin("plot3"),
        },
        BuiltinDocumentationLink {
            label: "surf",
            target: BuiltinDocumentationLinkTarget::Builtin("surf"),
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
            label: "Choosing the right plot type",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type"),
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
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/zlabel.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/zlabel/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/zlabel.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
