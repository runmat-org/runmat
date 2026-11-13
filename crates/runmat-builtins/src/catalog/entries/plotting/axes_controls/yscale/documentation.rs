use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("yscale"),
    slug: Some("yscale"),
    summary: "Query or set the Y-axis scale for the current axes or a specified axes handle.",
    description: "`yscale` controls whether the Y axis is displayed on a linear or logarithmic scale. It supports query, explicit scale assignment, command-form style string arguments, and explicit axes-target forms.",
    keywords: &["yscale", "y axis scale", "log scale", "axes"],
    related: &["xscale", "semilogy", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: scale = yscale()",
            "Syntax: yscale(scale)",
            "Syntax: yscale(ax, scale)",
            "scale ('linear' | 'log'): Scale value to apply to the Y axis.",
            "ax (axes handle): Target axes handle. If omitted, the current axes are used.",
            "`yscale` shares state with the axes `YScale` property, so `set(gca, 'YScale', 'log')` and `yscale('log')` are equivalent.",
            "Passing an explicit axes handle updates that axes without changing the current axes selection.",
            "`yscale` reads and writes axes metadata. The renderer applies log mapping in its direct GPU coordinate path without gathering GPU-resident plot data.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Apply scale to a specific subplot: % Only the target axes receives logarithmic Y scaling",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Switch the Y axis to logarithmic scale", program: "plot(1:100, (1:100).^2);\nyscale('log');\nyscale()", display_output: Some("ans = 'log'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(yscale(),'log'));" } },
        BuiltinExample { id: "example-2", title: "Apply scale to a specific subplot", program: "ax = subplot(1, 2, 2);\nplot(ax, 1:100, exp((1:100)/20));\nyscale(ax, 'log');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(yscale(ax),'log'));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "xscale", target: BuiltinDocumentationLinkTarget::Builtin("xscale") },
        BuiltinDocumentationLink { label: "semilogy", target: BuiltinDocumentationLinkTarget::Builtin("semilogy") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/yscale.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "yscale behavior and validation tests", location: "builtins::plotting::yscale::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
