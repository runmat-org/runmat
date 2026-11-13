use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("zoom"),
    slug: Some("zoom"),
    summary: "Control zoom mode and programmatic axes zooming for figures and axes.",
    description: "`zoom` toggles figure zoom mode, applies positive numeric zoom factors to axes limits, records and restores zoom baselines, and returns a MATLAB-compatible zoom handle object for `get`/`set` workflows.",
    keywords: &[
        "zoom",
        "axes zoom",
        "figure zoom",
        "interactive plotting",
    ],
    related: &[
        "xlim",
        "ylim",
        "get",
        "set",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "zoom",
                "zoom(option)",
                "zoom(factor)",
                "zoom(fig, option)",
                "zoom(ax, option)",
                "zoom(ax, factor)",
                "z = zoom",
                "z = zoom(fig)",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Arguments",
            paragraphs: &[
                "option",
                "'on' | 'off' | 'toggle' | 'xon' | 'yon' | 'reset' | 'out'",
                "Zoom command. `xon` and `yon` constrain zooming to one axis, `reset` records the current limits as the baseline, and `out` restores that baseline.",
                "factor",
                "positive numeric scalar",
                "Zoom factor applied around the current axes center. Values greater than 1 zoom in; values between 0 and 1 zoom out. Integer scalars from any native integer class are accepted.",
                "fig",
                "figure handle",
                "Target figure handle.",
                "ax",
                "axes handle",
                "Target axes handle. Explicit axes updates do not change the current axes selection. RunMat mode additionally accepts a typed-integer alias for an encoded handle.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Usage notes",
            paragraphs: &[
                "Statement-form `zoom()` toggles zoom mode. Output-form `z = zoom()` returns a zoom handle object.",
                "In output context, `z = zoom(fig)` returns the zoom object for `fig`; in statement context, `zoom(factor)` applies a numeric zoom factor.",
                "The zoom object supports live `get`/`set` for `Enable`, `Motion`, `Direction`, `RightClickAction`, `UseLegacyExplorationModes`, and callback/context-menu placeholder properties.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`zoom` mutates plotting metadata only. Numeric zoom factors update stored axes limits directly and do not gather or copy GPU-resident plot data.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Zoom in around the current axes center",
            program: "plot(1:10, rand(1, 10));\nbefore = xlim();\nzoom(2);\nafter = xlim();",
            display_output: Some("% Current axes limits shrink by a factor of 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(abs((after(2)-after(1))*2 - (before(2)-before(1))) < 1e-10); assert(abs(sum(after)-sum(before)) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use the zoom object with get and set",
            program: "plot(1:10);\nz = zoom;\nset(z, 'Enable', 'on', 'Motion', 'horizontal');\nget(z, 'Motion')",
            display_output: Some("ans = 'horizontal'"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(z, \"Enable\"), \"on\")); assert(strcmp(get(z, \"Motion\"), \"horizontal\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "xlim",
            target: BuiltinDocumentationLinkTarget::Builtin("xlim")
        },
        BuiltinDocumentationLink {
            label: "ylim",
            target: BuiltinDocumentationLinkTarget::Builtin("ylim")
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
                label: "crates/runmat-runtime/src/builtins/plotting/ops/zoom.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/zoom.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::plotting::zoom::tests",
            },
        ],
        notes: &[],
    },
    introduced: Some("0.5.3"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
