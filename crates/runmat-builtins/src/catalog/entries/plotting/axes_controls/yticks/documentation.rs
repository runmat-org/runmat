use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("yticks"),
    slug: Some("yticks"),
    summary: "Query or set Y-axis tick values.",
    description: "`yticks` returns or updates the tick mark locations for the Y axis of the current or specified axes.",
    keywords: &["yticks", "y-axis ticks", "axes", "plot styling"],
    related: &["xticks", "ylim", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "`yticks()` returns the current Y-axis tick values as a row vector.",
            "`yticks(ticks)` sets manual Y-axis tick values. All native integer classes are accepted and ordered exactly before conversion; adjacent values must remain distinct in graphics coordinates. `[]` removes Y-axis tick marks.",
            "`yticks('auto')` returns the Y tick mode to automatic tick generation. `yticks('manual')` freezes the current tick values.",
            "`yticks('mode')` returns `'auto'` or `'manual'`.",
            "`yticks(ax, ...)` applies the same forms to the specified axes handle. RunMat mode additionally accepts a typed-integer alias for an encoded axes handle.",
            "`get(ax, 'YTick')`, `set(ax, 'YTick', ticks)`, `get(ax, 'YTickMode')`, and `set(ax, 'YTickMode', mode)` share the same axes metadata.",
            "`yticks` only reads and writes axes metadata. It does not gather GPU-resident plot data; automatic ticks are generated from the axes display bounds, including explicit limits and cached plotted-data bounds.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Set nonuniform Y-axis ticks: % Shows tick marks at the requested Y coordinates",
            "Remove Y-axis tick marks: % Hides Y-axis tick marks",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set nonuniform Y-axis ticks", program: "plot(-5:5, (-5:5).^2);\nyticks([0 2 4 9 16 25]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(yticks(),[0 2 4 9 16 25]));" } },
        BuiltinExample { id: "example-2", title: "Remove Y-axis tick marks", program: "plot(1:10);\nyticks([]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isempty(yticks()));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "xticks", target: BuiltinDocumentationLinkTarget::Builtin("xticks") },
        BuiltinDocumentationLink { label: "ylim", target: BuiltinDocumentationLinkTarget::Builtin("ylim") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/yticks.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "yticks behavior and validation tests", location: "builtins::plotting::yticks::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
