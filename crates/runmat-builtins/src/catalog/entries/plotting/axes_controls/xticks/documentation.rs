use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xticks"),
    slug: Some("xticks"),
    summary: "Query or set X-axis tick values.",
    description: "`xticks` returns or updates the tick mark locations for the X axis of the current or specified axes.",
    keywords: &["xticks", "x-axis ticks", "axes", "plot styling"],
    related: &["yticks", "xlim", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "`xticks()` returns the current X-axis tick values as a row vector.",
            "`xticks(ticks)` sets manual X-axis tick values. All native integer classes are accepted and ordered exactly before conversion; adjacent values must remain distinct in graphics coordinates. `[]` removes X-axis tick marks.",
            "`xticks('auto')` returns the X tick mode to automatic tick generation. `xticks('manual')` freezes the current tick values.",
            "`xticks('mode')` returns `'auto'` or `'manual'`.",
            "`xticks(ax, ...)` applies the same forms to the specified axes handle. RunMat mode additionally accepts a typed-integer alias for an encoded axes handle.",
            "`get(ax, 'XTick')`, `set(ax, 'XTick', ticks)`, `get(ax, 'XTickMode')`, and `set(ax, 'XTickMode', mode)` share the same axes metadata.",
            "`xticks` only reads and writes axes metadata. It does not gather GPU-resident plot data; automatic ticks are generated from the axes display bounds, including explicit limits and cached plotted-data bounds.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Set nonuniform X-axis ticks: % Shows tick marks at the requested X coordinates",
            "Reset X-axis ticks to automatic mode: % Restores automatic X tick generation",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set nonuniform X-axis ticks", program: "plot(-5:5, (-5:5).^2);\nxticks([-5 -2.5 0 1 2.5 5]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(xticks(),[-5 -2.5 0 1 2.5 5]));" } },
        BuiltinExample { id: "example-2", title: "Reset X-axis ticks to automatic mode", program: "xticks([0 5 10]);\nxticks('auto');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xticks('mode'),'auto'));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "yticks", target: BuiltinDocumentationLinkTarget::Builtin("yticks") },
        BuiltinDocumentationLink { label: "xlim", target: BuiltinDocumentationLinkTarget::Builtin("xlim") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xticks.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xticks behavior and validation tests", location: "builtins::plotting::xticks::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
