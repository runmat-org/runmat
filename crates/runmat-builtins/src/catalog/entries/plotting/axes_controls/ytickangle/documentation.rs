use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ytickangle"),
    slug: Some("ytickangle"),
    summary: "Query or set Y-axis tick label rotation.",
    description: "`ytickangle` returns or updates the Y-axis tick label rotation in degrees for the current axes or specified axes handles.",
    keywords: &["ytickangle", "tick angle", "tick label rotation", "plotting", "axes"],
    related: &["xtickangle", "yticklabels", "yticks"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "`ytickangle()` returns the current Y tick label rotation, defaulting to `0`.",
            "`ytickangle(angle)` sets the current axes Y tick label rotation in degrees. Integer scalars from any native integer class are accepted.",
            "`ytickangle(ax)` queries a scalar axes handle without changing the current axes.",
            "`ytickangle(ax, angle)` accepts a scalar axes handle or numeric array of axes handles and sets each target. RunMat mode additionally accepts typed-integer aliases for encoded axes handles.",
            "The function shares state with `YTickLabelRotation` on axes and `TickLabelRotation` on the Y-axis ruler returned by `get(ax, 'YAxis')`.",
            "Signature: angle = ytickangle()",
            "Signature: ytickangle(angle)",
            "Signature: angle = ytickangle(ax)",
            "Signature: ytickangle(ax, angle)",
            "`ytickangle` only reads and writes axes metadata. It does not inspect, gather, or transform GPU-resident plot data.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Examples are verified by their resulting graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Runtime plotting bindings require the plot-core feature.",
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Rotate Y tick labels", program: "plot(1:3);\nytickangle(25);\nytickangle()", display_output: Some("ans = 25"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(ytickangle()==25);" } },
        BuiltinExample { id: "example-2", title: "Target a specific axes", program: "ax = subplot(1, 2, 2);\nytickangle(ax, -45);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(ytickangle(ax)==-45);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "xtickangle", target: BuiltinDocumentationLinkTarget::Builtin("xtickangle") },
        BuiltinDocumentationLink { label: "yticklabels", target: BuiltinDocumentationLinkTarget::Builtin("yticklabels") },
        BuiltinDocumentationLink { label: "yticks", target: BuiltinDocumentationLinkTarget::Builtin("yticks") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/ytickangle.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "ytickangle behavior and validation tests", location: "builtins::plotting::ytickangle::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "runmat-vm plotting::tickangle_dispatches_and_round_trips_properties" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
