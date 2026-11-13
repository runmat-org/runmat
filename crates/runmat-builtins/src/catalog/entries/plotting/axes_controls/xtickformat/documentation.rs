use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xtickformat"),
    slug: Some("xtickformat"),
    summary: "Query or set the X-axis tick label format for the current axes, a specified axes handle, or an axes array.",
    description: "`xtickformat` controls automatic X-axis tick label formatting. It supports query, numeric format assignment, MATLAB shorthand formats such as `usd` and `percentage`, custom printf-style numeric formats, date/duration format strings for compatibility, scalar axes query forms, and axes-array setter forms.",
    keywords: &["xtickformat", "x tick format", "tick label format", "axes"],
    related: &["ytickformat", "xticklabels", "xticks", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: fmt = xtickformat()",
            "Syntax: xtickformat(fmt)",
            "Syntax: fmt = xtickformat(ax)",
            "Syntax: xtickformat(ax, fmt)",
            "fmt (string scalar | character vector): Tick label format. `auto` maps to `%g`; shorthands include `usd`, `eur`, `gbp`, `jpy`, `degrees`, and `percentage`; custom numeric formats use printf-style conversions such as `%.2f` or `$%,.0f`.",
            "ax (axes handle | axes handle array): Target axes handle or axes handle array. Query forms require a scalar axes handle; setter forms apply to every axes in the array. If omitted, the current axes are used.",
            "`xtickformat` shares state with the X-axis ruler `TickLabelFormat` property, so `set(get(gca, 'XAxis'), 'TickLabelFormat', fmt)` and `xtickformat(fmt)` are equivalent for the current axes.",
            "Explicit `xticklabels` values take precedence over automatic formatting until tick label mode is returned to automatic.",
            "Date and duration format strings are stored and returned for compatibility. Current RunMat plotting axes are numeric, so nonnumeric date/duration rendering falls back to numeric label formatting until datetime/duration axes are introduced.",
            "`xtickformat` only reads and writes axes metadata. It does not inspect, gather, or transform GPU-resident plot data; formatted labels are generated from tick metadata at render/query time.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Apply a format to a specific subplot: % Only ax2 receives the dollar tick-label format",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Show X-axis tick labels with units", program: "plot(1:3, [10 20 30]);\nxticks([1 2 3]);\nxtickformat('%.1f s');\nxticklabels()", display_output: Some("{'1.0 s', '2.0 s', '3.0 s'}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xtickformat(),'%.1f s'));" } },
        BuiltinExample { id: "example-2", title: "Apply a format to a specific subplot", program: "ax1 = subplot(1, 2, 1);\nplot(ax1, 1:3);\nax2 = subplot(1, 2, 2);\nplot(ax2, 10:10:30);\nxtickformat(ax2, 'usd');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xtickformat(ax2),\"usd\"));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ytickformat", target: BuiltinDocumentationLinkTarget::Builtin("ytickformat") },
        BuiltinDocumentationLink { label: "xticklabels", target: BuiltinDocumentationLinkTarget::Builtin("xticklabels") },
        BuiltinDocumentationLink { label: "xticks", target: BuiltinDocumentationLinkTarget::Builtin("xticks") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xtickformat.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xtickformat behavior and validation tests", location: "builtins::plotting::xtickformat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "runmat-vm plotting::tickformat_dispatches_and_updates_axes_properties" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
