use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("ribbon"),
 slug: Some("ribbon"),
 summary: "Create 3-D ribbon plots from columns of data.",
 description: concat!(
    "`ribbon(Y)` renders each column of `Y` as a 3-D surface strip. Optional `X` ",

    "coordinates control the row positions, and optional `width` controls the strip ",

    "width around each column center. The return value is a surface handle for ",

    "vector input or a row vector of surface handles for matrix input.",

),

 keywords: &["ribbon",
 "ribbon plot",
 "3-D ribbon",
 "surface strip",
 "matlab ribbon",
 "plotting"],
 related: &["surf",
 "mesh",
 "view",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`ribbon(Y)` uses implicit x coordinates `1:size(Y,1)` and creates one surface ",

    "ribbon per column of `Y`.",

),
 "`ribbon(Y,width)` uses the supplied positive finite ribbon width around each column center.",
 concat!(
    "`ribbon(X,Y)` and `ribbon(X,Y,width)` use `X` as the x coordinates for each row ",

    "of `Y`; `numel(X)` must match the number of rows in `Y`.",

),
 concat!(
    "All eight integer classes are accepted for `X`, `Y`, and `width`. Source ",

    "`XData` and `ZData` retain authoritative native storage on the surface object, ",

    "while the renderer materializes floating geometry only at its explicit graphics ",

    "boundary.",

),
 "Vector `Y` input, including row-vector input, creates a single ribbon surface.",
 concat!(
    "`NaN` height values are preserved in the surface `ZData`, allowing callers to ",

    "represent missing or interrupted ribbon geometry.",

),
 "The axes-target form `ribbon(ax,...)` renders into the supplied axes handle.",
 concat!(
    "Surface style name/value pairs such as `FaceAlpha`, `FaceColor`, `EdgeColor`, ",

    "`Shading`, `Colormap`, `DisplayName`, `Lighting`, and `Visible` are accepted ",

    "through the same surface style parser used by `surf`.",

),
 concat!(
    "Each returned object has type `surface`; `get` exposes the actual ",

    "coordinate-grid `XData`, `YData`, and `ZData` for each ribbon strip.",

),
 concat!(
    "Surface handles support setting `XData`, `YData`, `ZData`, `FaceAlpha`, ",

    "`FaceColor`, `EdgeColor`, `Shading`, `Colormap`, `DisplayName`, `Lighting`, ",

    "`FlattenZ`, and `Visible`; unsupported surface property names raise an error ",

    "instead of silently no-oping.",

),
 "`DisplayName` defaults to the empty string unless supplied by a name/value pair or later `set` call."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`ribbon` is a rendering sink. Input columns are materialized into coordinate ",

    "grids and rendered through the shared `runmat-plot` surface pipeline. Direct ",

    "provider-resident ribbon grid construction is deferred because the current ",

    "surface GPU fast path is axis/Z-buffer based, while ribbon requires full X/Y ",

    "coordinate grids per strip.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "ribbon declares no direct numeric provider precision list; host ",

    "generation/sampling or owner-aware gather and installed graphics rendering are ",

    "separate boundaries.",

)] },

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
    "The accepted forms are Y, Y/width, X/Y, X/Y/width, with optional leading axes ",

    "and shared surface styles. Y must be nonempty numeric vector or rank-two ",

    "matrix; row/column vectors make one strip, matrix columns make separate strips. ",

    "A scalar second positional operand selects width, so the parser does not ",

    "disambiguate every scalar-Y X/Y spelling. X must be a finite vector of the Y ",

    "row length; width is a positive finite real scalar, default 0.75.",

),
 concat!(
    "Source XData/ZData retain native storage; height NaNs remain source data. Only ",

    "the selected height column is gathered into a strip rather than widening the ",

    "whole Y payload for each column. Two duplicated rows form the surface width ",

    "around the one-based column centre. Native source selection is separate from ",

    "floating SurfacePlot coordinate grids. No direct resident full-grid ribbon ",

    "constructor exists. Complex/sparse and GPU operands are not newly certified; ",

    "ordinary CPU strips and surface styling remain implemented.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Create a ribbon plot from matrix columns",
 program: "Y = [1 4; 2 5; 3 6];\nh = ribbon(Y);",
 display_output: Some("% h is a vector of surface handles"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(size(h),[1 2])); observed=get(h(1),'ZData'); ",

    "expected=[Y(:,1),Y(:,1)]; assert(isequal(observed(:),expected(:))); ",

    "observed=get(h(2),'ZData'); expected=[Y(:,2),Y(:,2)]; ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use explicit x coordinates and width",
 program: "x = [10 20 30];\nY = [2; 3; 4];\nh = ribbon(x, Y, 0.5);\nget(h, 'XData')",
 display_output: Some("ans =\n    10    10\n    20    20\n    30    30"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(h,'XData'); expected=[x(:),x(:)]; ",

    "assert(isequal(observed(:),expected(:))); observed=get(h,'YData'); ",

    "expected=[0.75*ones(3,1),1.25*ones(3,1)]; ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Style ribbon surfaces",
 program: concat!(
    "Y = [1 3 2; 2 4 3; 3 5 4];\nh = ribbon(Y, 0.4, 'FaceAlpha', 0.7, 'DisplayName', ",

    "'bands');\nget(h(1), 'FaceAlpha')",

),
 display_output: Some("ans =\n    0.7000"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(numel(h)==3); assert(abs(get(h(1),'FaceAlpha')-0.7)<1e-6); ",

    "assert(strcmp(get(h(1),'DisplayName'),'bands')); observed=get(h(3),'ZData'); ",

    "expected=[Y(:,3),Y(:,3)]; assert(isequal(observed(:),expected(:))); ",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Does ribbon create a new plot primitive?",
 answer: concat!(
    "No. RunMat lowers each ribbon to a standard surface strip, so existing surface ",

    "rendering, scene serialization, and `get`/`set` behavior apply.",

) },

BuiltinDocumentationFaq { question: "What does the width argument mean?",
 answer: concat!(
    "The width is the total span of each strip around its column center. With the ",

    "default width `0.75`, the first ribbon spans y positions `0.625` through ",

    "`1.375`.",

) },

BuiltinDocumentationFaq { question: "Can ribbon preserve gpuArray residency?",
 answer: concat!(
    "Not yet. `surf` can preserve provider residency for separable X/Y axes plus a Z ",

    "buffer, but `ribbon` needs generated full coordinate grids for each strip. That ",

    "provider-resident grid construction belongs to the queued GPU fast-path audit ",

    "work.",

) }],

 links: &[BuiltinDocumentationLink { label: "surf",
 target: BuiltinDocumentationLinkTarget::Builtin("surf") },

BuiltinDocumentationLink { label: "mesh",
 target: BuiltinDocumentationLinkTarget::Builtin("mesh") },

BuiltinDocumentationLink { label: "view",
 target: BuiltinDocumentationLinkTarget::Builtin("view") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/ribbon.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/ribbon.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::ribbon::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "vm",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
