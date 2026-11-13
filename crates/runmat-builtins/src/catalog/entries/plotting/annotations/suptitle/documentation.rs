use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`suptitle(txt)` updates the current figure and returns the super-title text handle.",
            "`suptitle(fig, txt, ...)` targets an explicit figure handle when the handle exists.",
            "String scalars, char arrays, string arrays, and cell arrays can all be used to create multiline super titles.",
            "Text styling properties such as `FontSize`, `FontWeight`, `FontAngle`, `Color`, `Interpreter`, and `Visible` use the shared plotting text-property system.",
            "`suptitle` and `sgtitle` update the same figure-level title object.",
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
        program: "subplot(2, 1, 1);\nplot(1:10, rand(1, 10));\nsubplot(2, 1, 2);\nplot(1:10, rand(1, 10));\nsuptitle('Experiment Summary');",
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
        program: "subplot(1, 2, 1);\nimagesc(peaks(20));\nsubplot(1, 2, 2);\nimagesc(peaks(20));\nh = suptitle('Field Comparison', 'FontSize', 18, 'FontWeight', 'bold');\nget(h, 'Type')",
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
        program: "fig = figure(42);\nsubplot(1, 2, 1);\nplot(0:0.1:1, sin(0:0.1:1));\nsubplot(1, 2, 2);\nplot(0:0.1:1, cos(0:0.1:1));\nsuptitle(fig, 'Sine and Cosine');",
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
    title: Some("suptitle"),
    slug: Some("suptitle"),
    summary: "Set a centered figure-level title using the legacy `suptitle` alias for `sgtitle`.",
    description: "`suptitle` sets the same figure-level title as `sgtitle` and exists primarily for compatibility with older MATLAB subplot code. It supports current-figure and explicit-figure forms and returns a text handle.",
    keywords: &["suptitle", "sgtitle", "super title", "subplot title", "figure title", "matlab suptitle"],
    related: &["sgtitle", "subplot", "title", "get", "set"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is suptitle different from sgtitle?",
            answer: "In RunMat, `suptitle` and `sgtitle` update the same figure-level super title. `suptitle` is provided as a compatibility spelling for MATLAB code that uses the older helper name.",
        },
        BuiltinDocumentationFaq {
            question: "How is suptitle different from title?",
            answer: "`title` attaches text to the current axes, so each subplot can have its own panel title. `suptitle` attaches text to the figure as a whole and is drawn once above all panels.",
        },
        BuiltinDocumentationFaq {
            question: "Can I update the super title after creating it?",
            answer: "Yes. `suptitle` returns a text handle, so you can inspect or update it with `get` and `set`.\n\n```matlab\nh = suptitle('Initial Title');\nset(h, 'String', 'Updated Title', 'FontWeight', 'bold');\n```",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "sgtitle",
            target: BuiltinDocumentationLinkTarget::Builtin("sgtitle"),
        },
        BuiltinDocumentationLink {
            label: "subplot",
            target: BuiltinDocumentationLinkTarget::Builtin("subplot"),
        },
        BuiltinDocumentationLink {
            label: "title",
            target: BuiltinDocumentationLinkTarget::Builtin("title"),
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
            label: "Complete plotting guide",
            target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime graphics operation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/suptitle.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/suptitle/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/suptitle.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
