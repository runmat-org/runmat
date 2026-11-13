use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("image"),
 slug: Some("image"),
 summary: "Display indexed or truecolor images in MATLAB and RunMat.",
 description: concat!(
    "`image` displays indexed or truecolor data as a graphics object and returns an ",

    "image handle. Axes placement, handle properties, and rendering semantics follow ",

    "MATLAB behavior.",

),

 keywords: &["image",
 "image display",
 "truecolor image",
 "indexed image",
 "matlab image",
 "imshow style image"],
 related: &["imagesc",
 "colormap",
 "colorbar",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`image(C)` displays image data using implicit axes; `image(X, Y, C)` places the ",

    "image explicitly on the target axes.",

),
 "Indexed images and truecolor images both use the shared surface/image rendering architecture.",
 "The returned value is an image-handle object that works with `get` and `set`.",
 concat!(
    "`image` is object-oriented and placement-oriented; use `imagesc` when you ",

    "specifically want scaled matrix visualization with color mapping semantics.",

),
 concat!(
    "Dedicated GPU-backed paths cover both indexed and truecolor image inputs when ",

    "plotting-compatible buffers are available.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`image` uses the shared image/surface path and supports GPU-backed paths for ",

    "indexed and truecolor image inputs.",

),
 concat!(
    "`image` can render exported indexed or truecolor GPU buffers directly when a ",

    "compatible shared plotting context is installed. Current handle publication ",

    "still captures CData through owning-provider readback, and indexed host ",

    "fallback can gather that source again. The renderer-buffer path is therefore ",

    "not a zero-readback or exactly-one-gather guarantee. Export/readback/emission ",

    "failures propagate through their owning completion protocol rather than ",

    "authorizing replay.",

),
 concat!(
    "Indexed images and truecolor images both have dedicated GPU-aware rendering ",

    "paths in the image stack.",

),
 concat!(
    "Image handle behavior, axes placement, and replay/export semantics remain ",

    "aligned across GPU and fallback paths.",

)] },

BuiltinDocumentationSection { heading: "Declared source precision and class coverage",
 paragraphs: &["single",
 "double"] },

BuiltinDocumentationSection { heading: "Coordinate broadcasting",
 paragraphs: &["none"] },

BuiltinDocumentationSection { heading: "Runtime contract and boundaries",
 paragraphs: &[concat!(
    "Catalog default bindings retain the actual RuntimeContext and original rooted ",

    "BuiltinArguments across parsing, provider readback, delegated calls and ",

    "completion. Value-selected axes, properties and output counts remain Runtime ",

    "decisions. Constructor results are host double graphics handles, not numerical ",

    "GPU arrays; hist instead returns double counts/centres, plotyy up to three ",

    "distinct handle outputs, and plotmatrix up to five.",

),
 concat!(
    "Rendering is a fusion boundary and requires the installed graphics host/device. ",

    "A direct compatible renderer route is not an elementwise or reduction provider ",

    "kernel. Resident buffers retain their owning provider and existing ",

    "export/error/completion protocol; an export, readback, allocation, cancellation ",

    "or emission failure is an error, not permission to replay an attempted ",

    "operation. Source data and renderer caches have distinct lifetimes. No new ",

    "distributed chart implementation is declared.",

),
 concat!(
    "The existing canonical CoordinateWork allocation policy refuses Managed calls ",

    "before unowned chart scratch, metadata, readback or scene mutation. It checks ",

    "captured cancellation generation and policy before publication and on ",

    "completion; ingress failures are Never replay. This is an explicit missing ",

    "finite owner, not an allocation estimate, algorithm completion or L25 closure. ",

    "Provider-private allocations, renderer lifetime accounting, synchronous long ",

    "geometry/copy cooperation, measured throughput and physical presentation remain ",

    "independent qualification gates.",

),
 concat!(
    "Native Single/Double/integer source ownership is distinguished from explicit ",

    "binary64/f32 graphics conversion. The renderer does not promise exact ",

    "wide-integer pixel positions. Logical, complex, sparse, scalar, vector, matrix, ",

    "empty and higher-rank forms follow the identity-specific parser rather than a ",

    "universal implicit cast. Sparse/complex rejection is not an integer ",

    "not-applicable audit. Full MATLAB properties, browser/native parity and ",

    "empty-shape conformance are not inferred from migration or historical test ",

    "references.",

)] },

BuiltinDocumentationSection { heading: "Supported representations and remaining forms",
 paragraphs: &[concat!(
    "Forms are C or X/Y/C plus shared surface-style pairs and ",

    "CDataMapping=direct/scaled. C-only textual controls are separated by actual ",

    "string/character-row kind; explicit numeric X/Y/C remains unchanged. This owner ",

    "does not parse an independent leading-axes form. X/Y axis extents and source ",

    "grid geometry are validated by the existing image/surface owner. Empty GPU ",

    "extents now reject consistently rather than being silently changed to one.",

),
 concat!(
    "Indexed CData uses direct mapping by default: floating indices are one-based, ",

    "integer indices zero-based. Three-channel truecolor and separately gated ",

    "four-channel data normalize native integer ranges or clamp finite floating ",

    "values at the renderer boundary. Native CData is retained for get. ",

    "Complex/sparse data and full MATLAB image/axes property parity remain outside ",

    "this implementation; the generic surface-style grammar does not imply all image ",

    "object setters exist.",

),
 concat!(
    "Floating indexed and truecolor buffers can use compatible shared-device ",

    "exporters. Retained CData currently requires an eager owning-provider property ",

    "snapshot even on a direct renderer route. Indexed host fallback may read the ",

    "numeric source again; neither zero-readback nor exactly-one-readback is ",

    "promised. A prepared export/emission error propagates through its existing ",

    "completion owner instead of host replay.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Display a simple indexed image",
 program: "C = [1 2 3; 3 2 1; 2 3 1];\nimage(C);\ncolormap('jet');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=C; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(strcmp(get(witness(1),'CDataMapping'),'direct'));",

) } },

BuiltinExample { id: "retained-2",
 title: "Display a truecolor RGB image",
 program: "img = zeros(100, 100, 3);\nimg(:,:,1) = 1;\nimg(25:75,25:75,2) = 1;\nimage(img);",
 display_output: Some("% Displays a truecolor image object"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=img; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(isequal(size(get(witness(1),'CData')),[100 100 3]));",

) } },

BuiltinExample { id: "retained-3",
 title: "Position an image explicitly on axes and inspect the handle",
 program: "C = reshape(1:16, 4, 4);\nh = image([0 3], [10 40], C);\nget(h, 'Type')",
 display_output: Some("ans =\n    'image'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(h,'CData'); expected=C; assert(isequal(observed(:),expected(:))); ",

    "assert(strcmp(get(h,'Type'),'image')); observed=get(h,'XData'); ",

    "assert(observed(1)==0&&observed(end)==3); observed=get(h,'YData'); ",

    "assert(observed(1)==10&&observed(end)==40);",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "How do I display RGB data with image?",
 answer: concat!(
    "Pass an MxNx3 array where the third dimension holds red, green, and blue ",

    "channels. Values should be in `[0, 1]` for doubles or `[0, 255]` for ",

    "uint8.\n\n```matlab\nimg = zeros(100, 100, 3);\nimg(:,:,1) = 1;           % full ",

    "red\nimg(30:70, 30:70, 2) = 1; % green square\nimage(img);\n```",

) },

BuiltinDocumentationFaq { question: "When should I use image vs imagesc?",
 answer: concat!(
    "`image` displays pixel data as-is — either indexed colors through a colormap or ",

    "direct RGB truecolor. `imagesc` automatically scales a 2-D matrix so its min ",

    "and max span the full colormap range. If you're showing actual image pixels or ",

    "pre-indexed data, use `image`. If you're visualizing a matrix of computed ",

    "values as a heatmap, use `imagesc`.",

) },

BuiltinDocumentationFaq { question: "Why does my image appear upside down?",
 answer: concat!(
    "By default, `image` sets the y-axis to go from top to bottom (row 1 at the ",

    "top), which matches matrix indexing. If your coordinate system expects y to ",

    "increase upward, call `axis xy` after `image` to flip the axis direction. Use ",

    "`axis ij` to restore the default matrix-style orientation.",

) }],

 links: &[BuiltinDocumentationLink { label: "imagesc",
 target: BuiltinDocumentationLinkTarget::Builtin("imagesc") },

BuiltinDocumentationLink { label: "colormap",
 target: BuiltinDocumentationLinkTarget::Builtin("colormap") },

BuiltinDocumentationLink { label: "colorbar",
 target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") },

BuiltinDocumentationLink { label: "Choosing the right plot type",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },

BuiltinDocumentationLink { label: "Styling plots and axes",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },

BuiltinDocumentationLink { label: "Plot replay and export",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },

BuiltinDocumentationLink { label: "Complete plotting guide",
 target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/image.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/image.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::image::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "integration",
 location: "runmat-plot/tests/renderer_tests.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
