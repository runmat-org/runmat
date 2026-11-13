use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`xline(x)` adds a vertical reference line at each coordinate in `x`. Coordinates may use any native integer class and cross to the graphics coordinate domain after validation.",
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
            "% Adds a red dashed vertical reference line at x = 5",
            "% Adds three dotted vertical reference lines",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Add a dashed threshold line",
        program: "plot(1:10, rand(1, 10));\nh = xline(5, '--r', 'Threshold');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(get(h,'Value')==5); assert(strcmp(get(h,'Label'),'Threshold'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Add multiple reference lines",
        program: "plot(1:10, rand(1, 10));\nh = xline([2 5 8], ':');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions:
                "assert(numel(h)==3); assert(get(h(1),'Value')==2); assert(get(h(3),'Value')==8);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xline"),
    slug: Some("xline"),
    summary: "Draw vertical reference lines on the current axes.",
    description: "`xline` adds one or more vertical reference lines with optional labels, line specs, and name-value styling, following MATLAB-compatible plotting behavior.",
    keywords: &["xline", "vertical reference line", "constant line", "plot annotation"],
    related: &["yline", "plot"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[

    ],
    links: &[
        BuiltinDocumentationLink {
            label: "yline",
            target: BuiltinDocumentationLinkTarget::Builtin("yline"),
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
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xline.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/xline/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/xline.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
