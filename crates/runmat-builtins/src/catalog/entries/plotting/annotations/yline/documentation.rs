use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`yline(y)` adds a horizontal reference line at each coordinate in `y`. Coordinates may use any native integer class and cross to the graphics coordinate domain after validation.",
            "Reference lines append to the current axes without replacing existing plot content.",
            "Line spans follow the active axes limits, including later `xlim` and `ylim` changes.",
            "The returned handle supports `get` and `set` for `Value`, `Color`, `LineWidth`, `LineStyle`, `Label`, `LabelOrientation`, `DisplayName`, and `Visible`.",
            "Integer RGB components must be zero or one, integer line widths must be positive, and numeric `Visible` values must be exactly zero or one.",
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
            "% Adds a black horizontal reference line at y = 0",
            "% Adds a dotted labeled horizontal reference line",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Add a zero baseline",
        program: "plot(-5:5, (-5:5).^2 - 10);\nh = yline(0, 'k', 'LineWidth', 1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(get(h,'Value')==0); assert(get(h,'LineWidth')==1);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Label a baseline",
        program: "plot(1:10, rand(1, 10));\nh = yline(0.5, ':', 'Baseline', 'LabelOrientation', 'horizontal');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(get(h,'Value')==0.5); assert(strcmp(get(h,'Label'),'Baseline'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("yline"),
    slug: Some("yline"),
    summary: "Draw horizontal reference lines on current or specified axes with MATLAB-compatible styling forms.",
    description: "`yline` adds one or more horizontal reference lines with optional labels, line specs, and name-value styling, following MATLAB-compatible plotting behavior.",
    keywords: &["yline", "horizontal reference line", "constant line", "plot annotation"],
    related: &["xline", "plot"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[

    ],
    links: &[
        BuiltinDocumentationLink {
            label: "xline",
            target: BuiltinDocumentationLinkTarget::Builtin("xline"),
        },
        BuiltinDocumentationLink {
            label: "plot",
            target: BuiltinDocumentationLinkTarget::Builtin("plot"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/yline.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/yline/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/yline.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
