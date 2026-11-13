use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xticklabels"),
    slug: Some("xticklabels"),
    summary: "Query or set X-axis tick labels for the current axes or a specified axes handle.",
    description: "`xticklabels` controls the display text associated with X-axis tick marks. It supports query, explicit label assignment, automatic mode, manual mode, mode query, and explicit axes-target forms.",
    keywords: &["xticklabels", "x tick labels", "axes", "plot labels"],
    related: &["xticks", "yticklabels", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: labels = xticklabels()",
            "Syntax: xticklabels(labels)",
            "Syntax: mode = xticklabels(mode)",
            "Syntax: xticklabels(ax, labels)",
            "Syntax: mode = xticklabels(ax, 'mode')",
            "labels (cell array of character vectors | string array | char array | []): Labels to associate with the current X-axis ticks. Fewer labels than ticks are padded with empty character vectors; an empty cell or empty numeric array removes labels.",
            "mode ('auto' | 'manual' | 'mode'): 'auto' returns tick labels to automatic generation, 'manual' freezes current labels, and 'mode' queries the current label mode.",
            "ax (axes handle): Target axes handle. If omitted, the current axes are used.",
            "Setting explicit labels also freezes the X tick values at their current positions, matching MATLAB's XTickMode side effect.",
            "Automatic labels are formatted from the current X tick values and cached display bounds, so this builtin does not gather plot data from accelerated backends.",
            "`xticklabels` only reads and writes axes metadata. It does not gather GPU-resident plot data; automatic labels are generated from current tick metadata and cached display bounds.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Set X-axis tick values and labels: % The X-axis shows the three supplied labels",
            "Query and reset label mode: ans = 'manual'\n% Labels return to automatic numeric formatting",
            "Set labels on a specific axes: % Only the target axes receives the labels",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set X-axis tick values and labels", program: "plot(0:10, (0:10).^2);\nxticks([0 5 10]);\nxticklabels({'x = 0', 'x = 5', 'x = 10'});", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(xticklabels())>=3);" } },
        BuiltinExample { id: "example-2", title: "Query and reset label mode", program: "xticklabels({'A', 'B'});\nxticklabels('mode')\nxticklabels('auto');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xticklabels('mode'),'auto'));" } },
        BuiltinExample { id: "example-3", title: "Set labels on a specific axes", program: "ax = subplot(1, 2, 1);\nplot(ax, 1:3);\nxticks(ax, [1 2 3]);\nxticklabels(ax, {'one', 'two', 'three'});", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(xticklabels(ax))>=3);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "xticks", target: BuiltinDocumentationLinkTarget::Builtin("xticks") },
        BuiltinDocumentationLink { label: "yticklabels", target: BuiltinDocumentationLinkTarget::Builtin("yticklabels") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xticklabels.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xticklabels behavior and validation tests", location: "builtins::plotting::xticklabels::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
