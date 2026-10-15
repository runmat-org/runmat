use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ytickformat"),
    slug: Some("ytickformat"),
    summary: "Query or set the Y-axis tick label format for the current axes, a specified axes handle, or an axes array.",
    description: "`ytickformat` controls automatic Y-axis tick label formatting. It supports query, numeric format assignment, MATLAB shorthand formats such as `usd` and `percentage`, custom printf-style numeric formats, date/duration format strings for compatibility, scalar axes query forms, and axes-array setter forms.",
    keywords: &["ytickformat", "y tick format", "tick label format", "axes"],
    related: &["xtickformat", "yticklabels", "yticks", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: fmt = ytickformat()",
            "Syntax: ytickformat(fmt)",
            "Syntax: fmt = ytickformat(ax)",
            "Syntax: ytickformat(ax, fmt)",
            "fmt (string scalar | character vector): Tick label format. `auto` maps to `%g`; shorthands include `usd`, `eur`, `gbp`, `jpy`, `degrees`, and `percentage`; custom numeric formats use printf-style conversions such as `%.2f` or `$%,.0f`.",
            "ax (axes handle | axes handle array): Target axes handle or axes handle array. Query forms require a scalar axes handle; setter forms apply to every axes in the array. If omitted, the current axes are used.",
            "`ytickformat` shares state with the Y-axis ruler `TickLabelFormat` property, so `set(get(gca, 'YAxis'), 'TickLabelFormat', fmt)` and `ytickformat(fmt)` are equivalent for the current axes.",
            "Explicit `yticklabels` values take precedence over automatic formatting until tick label mode is returned to automatic.",
            "Date and duration format strings are stored and returned for compatibility. Current RunMat plotting axes are numeric, so nonnumeric date/duration rendering falls back to numeric label formatting until datetime/duration axes are introduced.",
            "`ytickformat` only reads and writes axes metadata. It does not inspect, gather, or transform GPU-resident plot data; formatted labels are generated from tick metadata at render/query time.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Show Y-axis tick labels as percentages: {'0%', '0.5%', '1%'}",
            "Apply a format to a specific subplot: % Only ax2 receives the dollar tick-label format",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Show Y-axis tick labels as percentages", program: "plot(1:3, [0 0.5 1]);\nyticks([0 0.5 1]);\nytickformat('percentage');\nyticklabels()", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(ytickformat(),'percentage'));" } },
        BuiltinExample { id: "example-2", title: "Apply a format to a specific subplot", program: "ax1 = subplot(1, 2, 1);\nplot(ax1, 1:3);\nax2 = subplot(1, 2, 2);\nplot(ax2, 10:10:30);\nytickformat(ax2, '$%,.2f');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(ytickformat(ax2),\"$%,.2f\"));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "xtickformat", target: BuiltinDocumentationLinkTarget::Builtin("xtickformat") },
        BuiltinDocumentationLink { label: "yticklabels", target: BuiltinDocumentationLinkTarget::Builtin("yticklabels") },
        BuiltinDocumentationLink { label: "yticks", target: BuiltinDocumentationLinkTarget::Builtin("yticks") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/ytickformat.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "ytickformat behavior and validation tests", location: "builtins::plotting::ytickformat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "runmat-vm plotting::tickformat_dispatches_and_updates_axes_properties" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
