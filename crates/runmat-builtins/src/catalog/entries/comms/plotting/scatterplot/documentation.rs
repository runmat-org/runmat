use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/scatterplot.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("scatterplot"),
    slug: Some("scatterplot"),
    summary: "Plot complex samples as constellation points.",
    description: "`scatterplot(x)` plots the real and imaginary components of complex samples against each other. Optional decimation, zero-based offset, marker style, and existing scatter figure follow the Communications Toolbox call forms.",
    keywords: &["scatterplot", "constellation diagram", "communications toolbox", "complex baseband", "qam", "psk", "scatter"],
    related: &["scatter", "plot", "real", "imag", "pskmod", "qammod"],
    sections: &[
        BuiltinDocumentationSection { heading: "Call forms and figure state", paragraphs: &[
            "scatterplot(x) draws every sample with a constellation marker. scatterplot(x,n) draws every nth sample. The offset form starts at zero-based offset, equivalent to x(offset+1:n:end). Marker is a LineSpec forwarded to scatter.",
            "The fifth argument scatfig is an existing Figure handle, not an axes handle. The plot uses equal x/y scaling and grid lines. The returned handle is the underlying scatter graphics handle. Plotting changes local figure state and may invoke figure observers.",
            "The MATLAB-compatible sample classes are single and double, with double integer-valued n and offset. RunMat mode also admits real fixed-width sample values when exactly representable in binary64 and typed-integer decimation controls. Strict MATLAB mode rejects those extensions before plotting.",
        ] },
        BuiltinDocumentationSection { heading: "GPU and execution", paragraphs: &[
            "scatterplot is a rendering sink and a fusion boundary. For a supported non-decimated GPU input, the provider can expose resident real/imaginary buffers to the shared GPU scatter renderer. Decimation or unsupported splitting gathers once, then uses the same scatter and axes behavior.",
            "Rendering requires a local user-interface context and is not transferable as a remote worker graphics handle. Numeric inputs are validated before figure selection; provider readback or rendering may suspend.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "qpsk", title: "Plot four QPSK symbols", program: "x = [1+1i;-1+1i;-1-1i;1-1i]; h = scatterplot(x);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h));" } },
        BuiltinExample { id: "decimation-marker", title: "Plot every fourth sample with a marker", program: "rx = randn(1000,1)+1i*randn(1000,1); h = scatterplot(rx,4,0,'gx');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h));" } },
        BuiltinExample { id: "existing-figure", title: "Add points to an existing scatter figure", program: "h1 = scatterplot([1+1i;1-1i]); h2 = scatterplot([-1+1i;-1-1i],1,0,'b.',gcf());", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h1)); assert(isgraphics(h2));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How does scatterplot differ from scatter?", answer: "scatterplot takes one complex sample array and derives its real and imaginary coordinates; scatter takes separate x and y coordinates. scatterplot also supports Communications Toolbox decimation and offset controls." },
        BuiltinDocumentationFaq { question: "Does scatterplot support gpuArray inputs?", answer: "Yes. A supported no-decimation input can be rendered from provider-resident real and imaginary buffers. Decimated or unsupported inputs gather once before rendering." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "scatter", target: BuiltinDocumentationLinkTarget::Builtin("scatter") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "real", target: BuiltinDocumentationLinkTarget::Builtin("real") },
        BuiltinDocumentationLink { label: "imag", target: BuiltinDocumentationLinkTarget::Builtin("imag") },
        BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
        BuiltinDocumentationLink { label: "GPU plotting and residency", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/gpu-plotting-and-residency") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Decimation, typed controls, marker, figure selection, and host rendering", location: "crates/runmat-runtime/src/builtins/plotting/ops/scatterplot.rs" }],
        notes: &["The direct GPU renderer, browser figure oracle, and packaging behavior require integrated product verification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
