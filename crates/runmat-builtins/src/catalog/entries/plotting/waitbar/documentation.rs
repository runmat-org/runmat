use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("waitbar"), slug: Some("waitbar"),
    summary: "Create or update a waitbar progress figure.",
    description: "`waitbar` creates a lightweight progress figure or updates an existing waitbar handle. RunMat stores progress and message state on the figure so scripts can keep MATLAB-style progress workflows without requiring a native modal UI.",
    keywords: &["waitbar", "progress", "figure", "ui"],
    related: &["figure", "get"],
    sections: &[
        BuiltinDocumentationSection { heading: "Syntax", paragraphs: &[
            "`h = waitbar(x)`",
            "`h = waitbar(x, msg)`",
            "`h = waitbar(x, h)`",
            "`h = waitbar(x, h, msg)`",
            "`h = waitbar(x, h, msg, Name, Value, ...)`",
        ] },
        BuiltinDocumentationSection { heading: "Arguments", paragraphs: &[
            "`x` is a numeric scalar progress fraction. Finite values are clamped into `[0, 1]`; typed-integer storage is a separately gated RunMat extension and must cross the UI metadata boundary exactly.",
            "`h` is an existing waitbar figure handle to update. Typed-integer numeric aliases are a separately gated RunMat extension.",
            "`msg` is a text scalar, string array, or cell array of text. Arrays are stored as newline-separated message lines.",
        ] },
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "New waitbars are regular RunMat figures with `NumberTitle` set off and `Name` set to `Waitbar` unless overridden with property pairs.",
            "`get(h, 'WaitbarProgress')`, `get(h, 'WaitbarMessage')`, and `get(h, 'Tag')` expose the stored progress state.",
            "`waitbar(x)` updates the current waitbar when one exists; otherwise it creates a new waitbar figure.",
            "`Tag` is a normal figure property for waitbars; new waitbars default to MATLAB's `TMWWaitbar` tag unless a `Tag` property is supplied.",
            "`CreateCancelBtn`, `Units`, and `WindowStyle` are accepted as compatibility no-ops because RunMat does not execute native UI callback code or native modal window behavior from waitbar creation.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "`waitbar` manages host UI metadata only. Resident numeric progress and handle values are rejected without provider access.",
            "Returned numeric handles identify local graphics state and do not transfer that state to workers or foreign runtimes.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "example-1", title: "Create and update a waitbar",
        program: "h = waitbar(0.25, 'Loading');\nwaitbar(0.75, h, 'Almost done');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(get(h,'WaitbarProgress') == 0.75); assert(strcmp(get(h,'WaitbarMessage'),'Almost done'));",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/waitbar.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Waitbar runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/waitbar.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Waitbar runtime tests", location: "builtins::plotting::waitbar::tests" }],
        notes: &["The legacy example's output was a descriptive comment, not program output; a figure-state oracle replaces that caption. Browser/Desktop presentation remains a separate execution gate."],
    },
    introduced: Some("0.5.3"), status: Some(BuiltinDocumentationStatus::Partial),
};
