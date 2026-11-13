use crate::*;

const PLOT_CORE: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::Any,
    engine: BuiltinExampleEngine::Default,
    compiler: &[],
    runtime: &[BuiltinRuntimeCapability::PlotCore],
    toolchain: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("print"),
    slug: Some("print"),
    summary: "Export a figure to a PNG file.",
    description: "`print` writes the active or selected figure as a PNG through the current filesystem provider. It supports MATLAB-style filename, `-dpng`, and resolution options. Other printer and vector-image devices are not available in this exporter.",
    keywords: &["print", "figure export", "png", "plotting", "save figure"],
    related: &["figure", "gcf", "plot", "saveas"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and options", paragraphs: &[
            "`print(filename)` and `print(filename,'-dpng')` export the current figure. `print(fig,filename,'-dpng')` chooses a figure explicitly. Device and filename text can occur in command-style order, including `print -dpng command_style_plot.png`.",
            "`-rN` changes the output dimensions relative to the default 800-by-600 canvas at 150 DPI; `-r0` selects that default. If repeated, the last resolution token wins. PNG is the only available device. Recognized PDF, EPS, and SVG requests fail with an unsupported-device error rather than producing PNG under another extension.",
            "A filename without `.png` receives that extension. A requested logical success output is a RunMat extension; MATLAB-compatible file-export forms are statement calls. Typed-integer figure aliases are also RunMat-only and must exactly identify a graphics handle.",
        ] },
        BuiltinDocumentationSection { heading: "Storage and execution", paragraphs: &[
            "The output is written through the selected filesystem authority. Native hosts can export to ordinary paths; browser hosts write to their installed virtual filesystem, from which the host can expose a download artifact. Export does not change figure data or the current directory.",
            "Filename and option operands can materialize on the host. Plot data retained by the scene may be rendered through the shared WGPU path and fall back to CPU rasterization when headless GPU initialization is unavailable. `print` itself is an I/O and fusion boundary, not a numeric GPU kernel; it does not change plot-data residency.",
            "A host-known figure is copied before operand materialization. A resident numeric figure handle is selected after its value materializes; that selected scene is then copied before rendering. Figure changes before resident-handle selection may appear in the export, while later changes do not alter the copied scene.",
            "The file-export call can suspend during rendering or publication. Failures retain the rendering or filesystem cause; a rejected device or malformed host option has no file effect.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "active-png", title: "Export the active figure to PNG", program: "x = 0:0.1:2*pi;\nplot(x,sin(x));\nprint('sine_plot','-dpng');\ninfo = imfinfo('sine_plot.png');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(info.Width == 800); assert(info.Height == 600); assert(strcmp(info.Format,'png'));" } },
        BuiltinExample { id: "resolution", title: "Export at a higher resolution", program: "x = 0:0.1:1;\nplot(x,x.^2,'LineWidth',2);\nprint('quadratic_plot','-dpng','-r300');\ninfo = imfinfo('quadratic_plot.png');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(info.Width == 1600); assert(info.Height == 1200);" } },
        BuiltinExample { id: "selected-figure", title: "Export a selected figure", program: "fig = figure();\nplot(1:5,[1 4 2 5 3]);\nprint(fig,'explicit_handle_plot.png','-dpng');\ninfo = imfinfo('explicit_handle_plot.png');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(info.Width == 800); assert(strcmp(info.Format,'png'));" } },
        BuiltinExample { id: "command-order", title: "Put the device option before the filename", program: "plot(1:4,[1 3 2 4]);\nprint -dpng command_style_plot.png\ninfo = imfinfo('command_style_plot.png');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(info.Width == 800); assert(strcmp(info.Format,'png'));" } },
        BuiltinExample { id: "status-extension", title: "Request the RunMat export status", program: "plot(1:3,[2 1 3]);\nok = print('status_plot','-dpng');\ninfo = imfinfo('status_plot.png');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(ok); assert(strcmp(info.Format,'png'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which formats does print support?", answer: "PNG is available via `-dpng` or a `.png` filename. PDF, EPS, and SVG device tokens are recognized but unavailable in the current exporter." },
        BuiltinDocumentationFaq { question: "Where is a browser export written?", answer: "To the selected virtual filesystem. The browser host may expose that file as a downloadable artifact." },
        BuiltinDocumentationFaq { question: "Does -r300 change plot data?", answer: "No. It changes the exported pixel dimensions only." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "gcf", target: BuiltinDocumentationLinkTarget::Builtin("gcf") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
        BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Print runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/print.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "PNG and control tests", location: "builtins::plotting::print::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pending resident scene and filesystem ownership", location: "builtins::plotting::print::pending_tests" },
        ],
        notes: &["Historical four programs, three FAQs, and five related links are preserved; browser and native rendering/artifact execution remains an integration gate."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
