use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("colororder"),
    slug: Some("colororder"),
    summary: "Set or query the plotting color order in MATLAB and RunMat.",
    description: "`colororder` controls the implicit series colors used by line-style plotting commands. Public numeric RGB inputs are single or double matrices in [0,1]. RunMat additionally supports a gated host-only integer RGB extension.",
    keywords: &["colororder", "color order", "plot colors", "axes palette", "matlab colororder"],
    related: &["plot", "colormap", "set"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "`colororder()` returns the current figure color order as an m-by-3 RGB triplet matrix.",
            "`colororder(colors)` sets the color order for the current figure and applies it to existing axes.",
            "`colororder(ax, colors)` sets the color order for the specified axes without changing the current axes.",
            "`colororder(fig, colors)` sets the color order for every axes in the specified figure.",
            "Numeric color arrays must be m-by-3 RGB triplet matrices with finite values in [0,1]. Public numeric classes are single and double; a single RGB triplet vector is accepted as one color.",
            "All eight integer classes are accepted only in RunMat mode and only when every native channel is exactly 0 or 1. Strict compatibility mode rejects integer RGB setters before conversion.",
            "Integer channels cross an explicit host f32 graphics-state boundary. Query output is always an m-by-3 host double matrix; MATLAB-compatible setter forms expose no assigned output, while the gated RunMat setter-output extension returns that matrix.",
            "Resident numeric color arrays are rejected without provider access.",
            "String and cell string color arrays accept MATLAB color names, short color tokens, and `#RGB` or `#RRGGBB` hex strings.",
            "Palette names include `default`, `gem`, `gem12`, `glow`, `glow12`, `sail`, `reef`, `meadow`, `dye`, and `earth`.",
            "Changing the color order recolors existing line/scatter/bar-family series cyclically and controls future implicit plot colors.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graphics state remains host-owned. Resident control values reject before provider access; resident plotted arrays retain the rendering pipeline.",
            "Whole-call resource accounting, cooperative large host controls, and native/browser presentation require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set a custom RGB color order caption: % Existing and future implicit series use red, green, and blue cyclically.", program: "x = 0:0.1:1;\nplot(x, x, x, x.^2, x, x.^3);\ncolororder([1 0 0; 0 0.5 0; 0 0 1]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(colororder(),[1 0 0;0 0.5 0;0 0 1]));" } },
        BuiltinExample { id: "example-2", title: "Use color names and hex strings", program: "colororder({'red', '#00AAFF', 'black'});\nplot(1:5, rand(5,3));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(size(colororder(),2)==3);" } },
        BuiltinExample { id: "example-3", title: "Set an axes-specific color order", program: "ax1 = subplot(1,2,1);\nplot(rand(5,3));\nax2 = subplot(1,2,2);\nplot(rand(5,3));\ncolororder(ax2, 'gem');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(size(colororder(ax2),2)==3);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is colororder different from colormap?", answer: "`colororder` controls the discrete sequence of implicit colors for plot series such as lines and scatter groups. `colormap` controls continuous color mapping for image, surface, heatmap, and contour data." },
        BuiltinDocumentationFaq { question: "Does colororder gather gpuArray plot data?", answer: "No. The builtin only updates axes and figure style metadata plus existing plot object colors. It does not read or materialize plotted numeric data." },
        BuiltinDocumentationFaq { question: "Does setting colororder change the current axes?", answer: "No. Targeted forms such as `colororder(ax, colors)` update that axes without selecting it as the current axes." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/colororder.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "colororder behavior and validation tests", location: "builtins::plotting::colororder::tests" }],
        notes: &["Legacy behaviors, examples, explanatory output captions, and links are retained. Browser/native presentation remains execution-gated."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
