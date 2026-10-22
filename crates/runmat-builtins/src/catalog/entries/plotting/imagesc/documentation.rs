use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("imagesc"),
 slug: Some("imagesc"),
 summary: "Display scaled matrix images in MATLAB and RunMat.",
 description: concat!(
    "`imagesc` displays matrix data using scaled colormap mapping and returns an ",

    "image handle. Color-limit behavior and axes integration follow MATLAB semantics.",

),

 keywords: &["imagesc",
 "scaled image",
 "heatmap",
 "matrix visualization",
 "matlab imagesc",
 "colormap image"],
 related: &["image",
 "colorbar",
 "colormap",
 "axis"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &["`imagesc(C)` displays a matrix using implicit axes and scaled color mapping.",
 concat!(
    "`imagesc(X, Y, C)` places the scaled image explicitly on the axes using the ",

    "provided coordinate extents.",

),
 concat!(
    "Unlike `image`, `imagesc` is fundamentally about colormap-driven value ",

    "visualization rather than direct indexed/truecolor graphics-object placement.",

),
 "The returned value is still an image-handle object in the shared plotting handle system.",
 "`imagesc` works naturally with `colormap`, `colorbar`, `caxis`, and subplot-local axes state."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`imagesc` uses the same modern image/surface path as `image`, but applies ",

    "scaled color-mapped visualization semantics for matrix data.",

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
    "C or X/Y/C uses the same retained image parser and native classification as ",

    "image, with scaled mapping by default. CDataMapping may override that default. ",

    "A sole remaining two-element numeric control is a finite strictly increasing ",

    "explicit color range; otherwise shared surface-style pairs apply. Scalar-handle ",

    "output and retained CData class/shape are independent of rendered color ",

    "conversion.",

),
 concat!(
    "Explicit limits update the active axes color limits. Without explicit limits ",

    "the image builder uses the current axes snapshot when one exists; data-bound ",

    "color scaling in rendering is not a guarantee that every MATLAB automatic caxis ",

    "state transition is implemented. The same eager source snapshot, possible ",

    "indexed fallback readback, exporter failure ownership, RGBA extension and ",

    "finite metadata frontiers as image apply.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Display a matrix as a heatmap-style scaled image",
 program: concat!(
    "[X, Y] = meshgrid(linspace(-3, 3, 60), linspace(-3, 3, 60));\nA = sin(X) .* ",

    "cos(Y);\nimagesc(A);\ncolorbar;",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=A; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(strcmp(get(witness(1),'CDataMapping'),'scaled'));",

) } },

BuiltinExample { id: "retained-2",
 title: "Place a scaled image on explicit axes",
 program: "A = reshape(1:100, 10, 10);\nimagesc([-5 5], [0 1], A);\ncolormap('parula');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=A; ",

    "assert(isequal(observed(:),expected(:))); observed=get(witness(1),'XData'); ",

    "assert(observed(1)==-5&&observed(end)==5);",

) } },

BuiltinExample { id: "retained-3",
 title: "Use subplot-local color workflows",
 program: concat!(
    "[X, Y] = meshgrid(linspace(-3, 3, 40), linspace(-3, 3, 40));\nZ = sin(X) .* ",

    "cos(Y);\nsubplot(1, 2, 1);\nimagesc(Z);\ncolormap('jet');\ncolorbar;\nsubplot(1, 2, ",

    "2);\nimagesc(magic(20));\ncolormap('gray');",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)==1); ",

    "observed=get(witness,'CData'); expected=magic(20); ",

    "assert(isequal(observed(:),expected(:))); left=subplot(1,2,1); ",

    "witness=findobj(left,'Type','image'); assert(numel(witness)==1); ",

    "observed=get(witness,'CData'); expected=Z; ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-4",
 title: "Dense matrix as a heatmap",
 program: concat!(
    "[X, Y] = meshgrid(linspace(-pi, pi, 300), linspace(-pi, pi, 300));\nZ = sin(3*X) ",

    ".* cos(2*Y) + cos(X.*Y);\n\nimagesc(Z);\ncolormap('turbo');\ncolorbar;\ntitle('Scalar",

    " Field Heatmap');\nxlabel('Column');\nylabel('Row');\naxis equal;",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','image'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'CData'); expected=Z; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(isequal(size(get(witness(1),'CData')),[300 300]));",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "What's the difference between imagesc and image?",
 answer: concat!(
    "`imagesc` maps a 2-D matrix through the colormap with automatic scaling — the ",

    "min value maps to the bottom of the colormap and the max to the top. `image` ",

    "treats input as direct indexed or truecolor data without automatic scaling. Use ",

    "`imagesc` for heatmaps and value visualization; use `image` when you have ",

    "actual image pixel data (RGB arrays or pre-indexed values).",

) },

BuiltinDocumentationFaq { question: "How does colormap scaling work with imagesc?",
 answer: concat!(
    "Scaled image rendering maps data through the active colormap. Explicit ",

    "`imagesc(C,[lo hi])` limits update the active axes color limits; `caxis([lo ",

    "hi])` also provides explicit axes limits for subsequent image workflows. ",

    "Without explicit limits this owner snapshots any current axes limits and ",

    "otherwise lets the surface renderer use data-bound scaling; it does not certify ",

    "every MATLAB automatic caxis state transition.\n\n```matlab\nimagesc(A);\ncaxis([-1 ",

    "1]);\ncolorbar;\n```",

) },

BuiltinDocumentationFaq { question: "How do I display a matrix as a heatmap?",
 answer: concat!(
    "Pass the matrix directly to `imagesc` and add a colorbar for the legend. Pair ",

    "with a colormap that suits your data — diverging maps like `'coolwarm'` work ",

    "well for data centered around zero, sequential maps like `'parula'` for ",

    "positive-only ranges.\n\n```matlab\nimagesc(myMatrix);\ncolormap('parula');\ncolorbar",

    ";\n```",

) }],

 links: &[BuiltinDocumentationLink { label: "image",
 target: BuiltinDocumentationLinkTarget::Builtin("image") },

BuiltinDocumentationLink { label: "colorbar",
 target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },

BuiltinDocumentationLink { label: "colormap",
 target: BuiltinDocumentationLinkTarget::Builtin("colormap") },

BuiltinDocumentationLink { label: "axis",
 target: BuiltinDocumentationLinkTarget::Builtin("axis") },

BuiltinDocumentationLink { label: "Choosing the right plot type",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },

BuiltinDocumentationLink { label: "Styling plots and axes",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },

BuiltinDocumentationLink { label: "Plot replay and export",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },

BuiltinDocumentationLink { label: "Complete plotting guide",
 target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") }],

 media: &[BuiltinDocumentationMedia { kind: BuiltinDocumentationMediaKind::Image,
 url: "https://web.runmatstatic.com/builtin-image/runmat-matlab-plot-imagesc-scaler-field-heatmap.webp",
 alt: "Dense matrix as a heatmap" }],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/imagesc.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/imagesc.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::imagesc::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "integration",
 location: "runmat-plot/tests/renderer_tests.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
