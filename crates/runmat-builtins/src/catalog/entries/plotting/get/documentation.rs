use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("get"),
    slug: Some("get"),
    summary: "Get graphics object properties in MATLAB and RunMat.",
    description: "`get` reads properties from figure, axes, legend, text, and plot-child handles. The supported fields depend on the handle family. Graphics state remains local to the current runtime.",
    keywords: &["get", "graphics handle", "plot properties", "matlab get", "inspect plotting state"],
    related: &["set", "legend", "subplot", "plot"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`get(h)` returns a struct of the known properties for the selected graphics handle. `get(h, 'PropertyName')` returns one property value; unsupported properties raise a plotting error.",
                "Figure, axes, text, legend, and child plot objects expose different fields. Zoom, pan, and data-cursor control objects expose controls such as `Enable`, but do not expose a universal `Type` field. Axes-local fields include limits, scale modes, legend visibility, and text metadata. The returned property payload keeps its stored class, including integer data.",
                "Typed integer aliases for RunMat's numeric graphics handles require RunMat compatibility mode and exact representability. The documented compatibility surface is graphics objects, not arbitrary numeric arrays.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Source data and execution",
            paragraphs: &[
                "Graphics properties are host state, but line, scatter, and 3-D line source coordinates can be retained in plotting-owned GPU buffers. Reading those properties may suspend for source readback; it does not run a numeric array kernel or alter the caller's buffer ownership.",
                "A failed readback retains its typed cause. Native and browser execution use the same async runtime binding; an unknown handle or property still raises a plotting error.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Inspect a line property struct",
            program: "h = plot(1:5, [1 4 2 5 3]);\nprops = get(h);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(props.XData,1:5)); assert(isequal(props.YData,[1 4 2 5 3]));",
            },
        },
        BuiltinExample {
            id: "example-2", title: "Read an axes-local property",
            program: "ax = subplot(1, 2, 2);\nsemilogx(logspace(0, 2, 50), logspace(0, 2, 50));\nset(ax, 'FontSize', 12);\nget(ax, 'FontSize')",
            display_output: Some("ans =\n    12"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1, assertions: "assert(get(ax,'FontSize') == 12);",
            },
        },
        BuiltinExample {
            id: "example-3", title: "Inspect a legend handle",
            program: "plot(1:5, 1:5);\nlg = legend('series');\nget(lg, 'Orientation')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(lg,'Orientation'),'vertical'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How do I see available properties?", answer: "Call `get(h)` without a property name. The resulting struct lists the known properties for that handle type." },
        BuiltinDocumentationFaq { question: "Can I query axes handles?", answer: "Yes. Axes, legends, text objects, and plot-child handles use the same interface. For axes, `get(gca,'XScale')` reads the x-axis scale mode, `get(gca,'View')` reads the camera angle, and `get(gca,'FontSize')` reads the tick-label font size." },
        BuiltinDocumentationFaq { question: "What if a property does not exist?", answer: "RunMat raises a plotting error. Call `get(h)` to inspect the fields exposed by that handle family." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
        BuiltinDocumentationLink { label: "legend", target: BuiltinDocumentationLinkTarget::Builtin("legend") },
        BuiltinDocumentationLink { label: "subplot", target: BuiltinDocumentationLinkTarget::Builtin("subplot") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
        BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/get.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Async property readback", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/core/properties/async_readback.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Graphics property tests", location: "builtins::plotting::get::tests" }],
        notes: &["Legacy description, three programs, FAQs, related links and source attribution are retained. BrowserGraphics examples and typed readback error/abandonment still require integrated browser and native execution qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
