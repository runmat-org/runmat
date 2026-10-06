use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Text may be supplied as a string, character vector, string array, cell array, or numeric scalar.",
            "Native integer scalar labels use their exact decimal spelling.",
            "`FontSize` accepts single, double, and all eight native integer classes; values must be positive and finite.",
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
        program: "plot(1:10, rand(1, 10));\nylabel('Amplitude')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'YLabel'),'String'),'Amplitude'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Use an integer label and font size",
        program: "ylabel(intmax('uint64'), 'FontSize', uint8(14))",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(get(gca,'YLabel'),'String'),'18446744073709551615'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ylabel"),
    slug: Some("ylabel"),
    summary: "Set the y-axis label for the current or specified axes.",
    description: "`ylabel(txt)` sets the y-axis label and returns the label object's graphics handle.",
    keywords: &["ylabel", "axes label", "plot annotation"],
    related: &["xlabel", "zlabel"],
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
            label: "xlabel",
            target: BuiltinDocumentationLinkTarget::Builtin("xlabel"),
        },
        BuiltinDocumentationLink {
            label: "zlabel",
            target: BuiltinDocumentationLinkTarget::Builtin("zlabel"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/ylabel.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/ylabel/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/ylabel.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
