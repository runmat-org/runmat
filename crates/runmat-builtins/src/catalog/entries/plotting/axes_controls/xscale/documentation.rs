use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xscale"),
    slug: Some("xscale"),
    summary: "Query or set the X-axis scale for the current axes or a specified axes handle.",
    description: "`xscale` controls whether the X axis is displayed on a linear or logarithmic scale. It supports query, explicit scale assignment, command-form style string arguments, and explicit axes-target forms.",
    keywords: &["xscale", "x axis scale", "log scale", "axes"],
    related: &["yscale", "semilogx", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Forms and behavior", paragraphs: &[
            "Syntax: scale = xscale()",
            "Syntax: xscale(scale)",
            "Syntax: xscale(ax, scale)",
            "scale ('linear' | 'log'): Scale value to apply to the X axis.",
            "ax (axes handle): Target axes handle. If omitted, the current axes are used.",
            "`xscale` shares state with the axes `XScale` property, so `set(gca, 'XScale', 'log')` and `xscale('log')` are equivalent.",
            "Passing an explicit axes handle updates that axes without changing the current axes selection.",
            "`xscale` reads and writes axes metadata. The renderer applies log mapping in its direct GPU coordinate path without gathering GPU-resident plot data.",
            "Introduced in RunMat 0.5.3.",
        ] },
        BuiltinDocumentationSection { heading: "Legacy example output notes", paragraphs: &[
            "Apply scale to a specific subplot: % Only the target axes receives logarithmic X scaling",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics controls are host-side state. Unsupported resident control operands reject without provider dispatch; plotted resident arrays retain their rendering owner.",
            "Whole-call allocation/cooperation and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Switch the X axis to logarithmic scale", program: "plot(1:1000, log10(1:1000));\nxscale('log');\nxscale()", display_output: Some("ans = 'log'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xscale(),'log'));" } },
        BuiltinExample { id: "example-2", title: "Apply scale to a specific subplot", program: "ax = subplot(1, 2, 2);\nplot(ax, 1:100, rand(1, 100));\nxscale(ax, 'log');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(xscale(ax),'log'));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "yscale", target: BuiltinDocumentationLinkTarget::Builtin("yscale") },
        BuiltinDocumentationLink { label: "semilogx", target: BuiltinDocumentationLinkTarget::Builtin("semilogx") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xscale.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xscale behavior and validation tests", location: "builtins::plotting::xscale::tests" }],
        notes: &["Legacy syntax, arguments, behavior, notes, example captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
