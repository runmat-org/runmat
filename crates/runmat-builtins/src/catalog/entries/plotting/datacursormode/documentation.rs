use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datacursormode"),
    slug: Some("datacursormode"),
    summary: "Control data cursor interaction mode for figures.",
    description: "`datacursormode` toggles or configures figure-level data cursor mode state. It returns a live data cursor manager object whose supported properties can be read or updated through `get` and `set`.",
    keywords: &[
        "datacursormode",
        "data cursor",
        "data tips",
        "interactive plotting",
    ],
    related: &[
        "dataTipTextRow",
        "get",
        "set",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "datacursormode",
                "datacursormode(option)",
                "datacursormode(fig, option) — RunMat numeric figure aliases only, requiring datacursormode-numeric-handle-alias; native Figure-object targets are not implemented.",
                "dcm = datacursormode",
                "dcm = datacursormode(fig) — RunMat numeric figure aliases only, requiring datacursormode-numeric-handle-alias; native Figure-object targets are not implemented.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Arguments",
            paragraphs: &[
                "option",
                "'on' | 'off' | 'toggle'",
                "Data cursor mode command.",
                "fig",
                "RunMat numeric figure alias (extension only; native Figure objects are not admitted)",
                "Target figure handle.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Usage notes",
            paragraphs: &[
                "Statement-form `datacursormode()` toggles data cursor mode. Output-form `dcm = datacursormode()` returns a data cursor manager object.",
                "The data cursor manager object supports `Enable`, `SnapToDataVertex`, `DisplayStyle`, `Interpreter`, `UpdateFcn`, and read-only `Figure` properties. `UpdateFcn` accepts an empty value or a function handle.",
                "RunMat numeric and typed-integer figure aliases require `datacursormode-numeric-handle-alias`. Native Figure-object admission is not implemented, and the manager's read-only `Figure` property currently exposes RunMat's numeric figure alias rather than a native Figure object.",
                "The undocumented `ok` result from command forms requires `datacursormode-status-output`.",
                "Broad numeric truthiness and yes/no aliases require `datacursormode-onoff-aliases`; strict Enable accepts on/off or scalar numeric/logical zero and one, while strict SnapToDataVertex accepts on/off text.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Limitations",
            paragraphs: &[
                "R2025a `datacursormode(ax, option)` is not yet implemented because RunMat currently stores data-cursor interaction mode per figure rather than per axes.",
                "The read-only `Figure` property is currently a numeric RunMat alias, not a native Figure object; numeric target forms are extension-only and are intentionally absent from the public signature metadata.",
                "UpdateFcn is stored on the manager object but is not yet invoked by graphics-host data-tip interaction callbacks.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`datacursormode` mutates plotting interaction metadata only. It does not gather or copy GPU-resident plot data.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Enable data cursor mode",
            program: "plot(1:10);\ndatacursormode on;",
            display_output: Some("% Data cursor mode is enabled"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "dcm = datacursormode(); assert(strcmp(get(dcm, \"Enable\"), \"on\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use the data cursor manager object",
            program: "plot(1:10);\ndcm = datacursormode;\nset(dcm, 'DisplayStyle', 'window', 'SnapToDataVertex', 'off');\nget(dcm, 'DisplayStyle')",
            display_output: Some("ans = 'window'"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(dcm, \"DisplayStyle\"), \"window\")); assert(strcmp(get(dcm, \"SnapToDataVertex\"), \"off\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "dataTipTextRow",
            target: BuiltinDocumentationLinkTarget::Builtin("dataTipTextRow")
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
                label: "crates/runmat-runtime/src/builtins/plotting/ops/datacursormode.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/datacursormode.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::plotting::datacursormode::tests",
            },
        ],
        notes: &[],
    },
    introduced: Some("0.5.3"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
