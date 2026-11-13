use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pan"),
    slug: Some("pan"),
    summary: "Control interactive axes panning for figures and axes.",
    description: "`pan` toggles or configures MATLAB-compatible figure panning state. It supports figure and axes targets, returns a live pan mode object, and integrates with `get` and `set` for common pan object properties.",
    keywords: &[
        "pan",
        "axes pan",
        "figure pan",
        "interactive plotting",
    ],
    related: &[
        "zoom",
        "get",
        "set",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "pan",
                "pan(option)",
                "pan(fig, option)",
                "pan(ax, option)",
                "p = pan",
                "p = pan(fig)",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Arguments",
            paragraphs: &[
                "option",
                "'on' | 'off' | 'toggle' | 'xon' | 'yon'",
                "Pan mode command. `xon` and `yon` enable panning constrained to the horizontal or vertical axis.",
                "fig",
                "figure handle",
                "Target figure handle.",
                "ax",
                "axes handle",
                "Target axes handle.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Usage notes",
            paragraphs: &[
                "Statement-form `pan()` toggles pan mode. Output-form `p = pan()` returns a pan handle object.",
                "The pan object supports `Enable`, `Motion`, `UIContextMenu`, `ButtonDownFilter`, `ActionPreCallback`, `ActionPostCallback`, and `FigureHandle` properties.",
                "`Enable` accepts exact numeric or logical scalar 0 and 1, including all native integer classes. Typed-integer numeric aliases for figure or axes objects are a separately gated RunMat graphics-handle extension.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`pan` mutates plotting interaction metadata only. It does not gather or copy GPU-resident plot data.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Enable horizontal panning",
            program: "plot(1:10);\npan xon;",
            display_output: Some("% Pan mode is enabled horizontally"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "p = pan(); assert(strcmp(get(p, \"Enable\"), \"on\")); assert(strcmp(get(p, \"Motion\"), \"horizontal\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use the pan object with get and set",
            program: "plot(1:10);\np = pan;\nset(p, 'Enable', 'on', 'Motion', 'vertical');\nget(p, 'Motion')",
            display_output: Some("ans = 'vertical'"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(p, \"Enable\"), \"on\")); assert(strcmp(get(p, \"Motion\"), \"vertical\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "zoom",
            target: BuiltinDocumentationLinkTarget::Builtin("zoom")
        },
        BuiltinDocumentationLink {
            label: "get",
            target: BuiltinDocumentationLinkTarget::Builtin("get")
        },
        BuiltinDocumentationLink {
            label: "set",
            target: BuiltinDocumentationLinkTarget::Builtin("set")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "crates/runmat-runtime/src/builtins/plotting/ops/pan.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/pan.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::plotting::pan::tests",
            },
        ],
        notes: &[],
    },
    introduced: Some("0.5.3"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
