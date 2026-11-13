use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xtickangle"),
    slug: Some("xtickangle"),
    summary: "Query or set X-axis tick label rotation.",
    description: "`xtickangle` returns or updates the X-axis tick label rotation in degrees for the current axes or specified axes handles.",
    keywords: &["xtickangle", "tick angle", "tick label rotation", "plotting", "axes"],
    related: &["ytickangle", "xticklabels", "xticks"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "`xtickangle()` returns the current X tick label rotation, defaulting to `0`.",
            "`xtickangle(angle)` sets the current axes X tick label rotation in degrees. Integer scalars from any native integer class are accepted.",
            "`xtickangle(ax)` queries a scalar axes handle without changing the current axes.",
            "`xtickangle(ax, angle)` accepts a scalar axes handle or numeric array of axes handles and sets each target. RunMat mode additionally accepts typed-integer aliases for encoded axes handles.",
            "The function shares state with `XTickLabelRotation` on axes and `TickLabelRotation` on the X-axis ruler returned by `get(ax, 'XAxis')`.",
            "Signature: angle = xtickangle()",
            "Signature: xtickangle(angle)",
            "Signature: angle = xtickangle(ax)",
            "Signature: xtickangle(ax, angle)",
            "`xtickangle` only reads and writes axes metadata. It does not inspect, gather, or transform GPU-resident plot data.",
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
        BuiltinExample { id: "example-1", title: "Rotate X tick labels", program: "plot(1:3);\nxtickangle(45);\nxtickangle()", display_output: Some("ans = 45"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(xtickangle()==45);" } },
        BuiltinExample { id: "example-2", title: "Target a specific axes", program: "ax = subplot(1, 2, 1);\nxtickangle(ax, -30);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(xtickangle(ax)==-30);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ytickangle", target: BuiltinDocumentationLinkTarget::Builtin("ytickangle") },
        BuiltinDocumentationLink { label: "xticklabels", target: BuiltinDocumentationLinkTarget::Builtin("xticklabels") },
        BuiltinDocumentationLink { label: "xticks", target: BuiltinDocumentationLinkTarget::Builtin("xticks") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xtickangle.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xtickangle behavior and validation tests", location: "builtins::plotting::xtickangle::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "runmat-vm plotting::tickangle_dispatches_and_round_trips_properties" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
