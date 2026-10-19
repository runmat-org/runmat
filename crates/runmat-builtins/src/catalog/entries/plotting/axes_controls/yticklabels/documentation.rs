use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("yticklabels"),
    slug: Some("yticklabels"),
    summary: "Query or set Y-axis tick labels for the current axes or a specified axes handle.",
    description: "`yticklabels` controls the display text associated with Y-axis tick marks. It supports query, explicit label assignment, automatic mode, manual mode, mode query, and explicit axes-target forms.",
    keywords: &["yticklabels", "y tick labels", "axes", "plot labels"],
    related: &["yticks", "xticklabels", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: labels = yticklabels()",
            "Syntax: yticklabels(labels)",
            "Syntax: mode = yticklabels(mode)",
            "Syntax: yticklabels(ax, labels)",
            "Syntax: mode = yticklabels(ax, 'mode')",
            "labels (cell array of character vectors | string array | char array | []): Labels to associate with the current Y-axis ticks. Fewer labels than ticks are padded with empty character vectors; an empty cell or empty numeric array removes labels.",
            "mode ('auto' | 'manual' | 'mode'): 'auto' returns tick labels to automatic generation, 'manual' freezes current labels, and 'mode' queries the current label mode.",
            "ax (axes handle): Target axes handle. If omitted, the current axes are used.",
            "Setting explicit labels also freezes the Y tick values at their current positions, matching MATLAB's YTickMode side effect.",
            "Automatic labels are formatted from the current Y tick values and cached display bounds, so this builtin does not gather plot data from accelerated backends.",
            "`yticklabels` only reads and writes axes metadata. It does not gather GPU-resident plot data; automatic labels are generated from current tick metadata and cached display bounds.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Set Y-axis tick values and labels: % The Y-axis shows the three supplied labels",
            "Query and reset label mode: ans = 'manual'\n% Labels return to automatic numeric formatting",
            "Set labels on a specific axes: % Only the target axes receives the labels",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set Y-axis tick values and labels", program: "plot(0:10, (0:10).^2);\nyticks([0 50 100]);\nyticklabels({'low', 'mid', 'high'});", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(yticklabels())>=3);" } },
        BuiltinExample { id: "example-2", title: "Query and reset label mode", program: "yticklabels({'A', 'B'});\nyticklabels('mode')\nyticklabels('auto');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(yticklabels('mode'),'auto'));" } },
        BuiltinExample { id: "example-3", title: "Set labels on a specific axes", program: "ax = subplot(1, 2, 2);\nplot(ax, 1:3);\nyticks(ax, [1 2 3]);\nyticklabels(ax, {'one', 'two', 'three'});", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(yticklabels(ax))>=3);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "yticks", target: BuiltinDocumentationLinkTarget::Builtin("yticks") },
        BuiltinDocumentationLink { label: "xticklabels", target: BuiltinDocumentationLinkTarget::Builtin("xticklabels") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/yticklabels.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "yticklabels behavior and validation tests", location: "builtins::plotting::yticklabels::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
