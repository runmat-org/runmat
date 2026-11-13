use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Text may be supplied as a string, character vector, string array, or cell array of strings.",
            "A numeric scalar is converted to label text. Native integer scalars, including one-element integer arrays, use their exact decimal spelling.",
            "The `FontSize` property accepts single, double, and all eight native integer classes. Values must be positive, finite, and within the renderer's supported size range.",
            "An axes handle may be supplied first. RunMat mode additionally accepts a typed-integer alias for an encoded axes handle.",
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
        title: "Label an axis",
        program: "plot(1:10, rand(1, 10));\nxlabel('Time (s)')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'XLabel'),'String'),'Time (s)'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Use an integer label and font size",
        program: "xlabel(intmax('uint64'), 'FontSize', uint8(14))",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'XLabel'),'String'),'18446744073709551615'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xlabel"),
    slug: Some("xlabel"),
    summary: "Set the x-axis label for the current or specified axes.",
    description: "`xlabel(txt)` sets the x-axis label and returns the label object's graphics handle.",
    keywords: &["xlabel", "axes label", "plot annotation"],
    related: &["ylabel", "title"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[

    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB language extensions",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions"),
        },
        BuiltinDocumentationLink {
            label: "ylabel",
            target: BuiltinDocumentationLinkTarget::Builtin("ylabel"),
        },
        BuiltinDocumentationLink {
            label: "title",
            target: BuiltinDocumentationLinkTarget::Builtin("title"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xlabel.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/xlabel/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/xlabel.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
