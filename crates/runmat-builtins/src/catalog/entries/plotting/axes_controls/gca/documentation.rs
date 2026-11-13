use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gca"),
    slug: Some("gca"),
    summary: "Return the current axes handle in MATLAB and RunMat.",
    description: "`gca` returns the handle for the current axes in the active figure, or in a specific figure when a figure handle is supplied. Optional struct mode returns a snapshot of current axes context for diagnostics and tooling.",
    keywords: &["gca", "current axes", "axes handle", "plotting"],
    related: &["figure", "gcf", "subplot", "get", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "`gca()` returns a scalar numeric axes handle.",
            "`gca(fig)` returns the current axes handle for the specified figure handle.",
            "`gca('struct')` returns a struct with fields `handle`, `figure`, `rows`, `cols`, and `index`.",
            "The current axes reflect the active subplot when `subplot` has been used.",
            "The numeric axes handle can be passed to `get` and `set`, including common axes styling such as `FontSize`.",
            "Unsupported arguments raise `gca` plotting errors.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "`gca` returns host-side handles or structs without provider dispatch or numeric GPU kernels. It reads the plotting registry, creating the ordinary default figure when none exists.",
            "Graphics state remains host-owned. Resident control values reject before provider access; resident plotted arrays retain the rendering pipeline.",
            "Whole-call resource accounting, cooperative large host controls, and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Read the current axes handle caption: % ax is a numeric axes handle", program: "figure(1);\nax = gca()", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isnumeric(ax));" } },
        BuiltinExample { id: "example-2", title: "Inspect the active subplot layout", program: "subplot(2, 2, 3);\ninfo = gca(\"struct\")", display_output: Some("info =\n  struct with fields:\n    handle\n    figure\n    rows\n    cols\n    index"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(info.rows==2 && info.cols==2 && info.index==3);" } },
        BuiltinExample { id: "example-3", title: "Read the current axes of a specific figure caption: % ax1 is the current axes handle for figure 1", program: "f1 = figure(1);\nsubplot(2, 2, 3);\nf2 = figure(2);\nax1 = gca(f1)", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isnumeric(ax1));" } },
        BuiltinExample { id: "example-4", title: "Style the current axes", program: "plot(1:3, [1 2 3]);\nset(gca, 'FontSize', 10);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(get(gca,'FontSize')==10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is special about `gca('struct')`?", answer: "It is a RunMat extension that returns a struct snapshot of the active axes state instead of just the encoded numeric handle." },
        BuiltinDocumentationFaq { question: "Does `gca` reflect `subplot` selection?", answer: "Yes. When a subplot is active, `gca` reports that subplot's axes handle and metadata." },
        BuiltinDocumentationFaq { question: "Can `gca` inspect a non-current figure?", answer: "Yes. Pass a figure handle, for example `gca(fig)`, to read that figure's current axes without changing the active figure." },
        BuiltinDocumentationFaq { question: "Does `gca` run on the GPU?", answer: "No. It only reads host-side plotting state." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "gcf", target: BuiltinDocumentationLinkTarget::Builtin("gcf") },
        BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/gca.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "gca behavior and validation tests", location: "builtins::plotting::gca::tests" }],
        notes: &["Legacy behaviors, examples, explanatory output captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
