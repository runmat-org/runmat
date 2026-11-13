use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Coordinates can use single, double, or any of the eight integer classes.",
            "The `Position` property retains the coordinate class, shape, and exact values. Rendering uses a separate floating-point geometry cache.",
            "A two-element `Position` value is stored with a third coordinate of zero in the same numeric class.",
            "The returned graphics handle supports `get` and `set` for `String`, `Position`, `FontWeight`, `FontAngle`, `Interpreter`, `Color`, `FontSize`, and `Visible`.",
            "Typed-integer numeric axes aliases and explicit gpuArray coordinates are RunMat-mode extensions.",
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
        title: "Label a point on a curve",
        program: "plot(1:5, [2 4 3 5 4]);\nh = text(4, 5, \"peak\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'peak')); assert(isequal(get(h,'Position'),[4 5 0]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Add a 3-D annotation",
        program: "plot3(1:3, 2:4, 5:7);\nh = text(2, 3, 6, \"sample\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(h,'String'),'sample')); assert(isequal(get(h,'Position'),[2 3 6]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Use integer coordinates",
        program: "h = text(int64(12), int64(-3), \"checkpoint\");\nposition = get(h, \"Position\")",
        display_output: Some("position = 1×3 int64\n    12    -3     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isa(position,'int64')); assert(isequal(position(1:2),int64([12 -3])));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("text"),
    slug: Some("text"),
    summary: "Add text at a position in a 2-D or 3-D plot.",
    description: "`text(x,y,txt)` adds an annotation at a 2-D position. Include `z` for a 3-D position, or provide an axes handle first to select the target axes.",
    keywords: &["text", "annotation", "label", "plotting"],
    related: &["plot", "title"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are integer positions converted to double?",
            answer: "No. The graphics property keeps native integer storage and exact values. Only the derived coordinates sent to the renderer are converted.",
        },
        BuiltinDocumentationFaq {
            question: "Can `text` label several points in one call?",
            answer: "Not yet. Each call currently accepts one scalar 2-D or 3-D position; vector coordinates and per-point label arrays remain unsupported.",
        },
        BuiltinDocumentationFaq {
            question: "Can I pass resident coordinates?",
            answer: "Automatically resident coordinates are gathered transparently through their owning provider. Explicit gpuArray coordinates require RunMat mode because the current public compatibility target does not document an interactive GPU form for `text`.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "plot",
            target: BuiltinDocumentationLinkTarget::Builtin("plot"),
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
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/text.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed identity contract",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/plotting/annotations/text/contract.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Graphics scene and handle behavior",
            location: "crates/runmat-runtime/src/builtins/plotting/ops/text.rs::tests",
        }],
        notes: &["Catalog source declaration and legacy runtime unit tests are not evidence of integrated native, browser, graphics-provider or placement execution."],
    },
    introduced: None,
    status: None,
};
