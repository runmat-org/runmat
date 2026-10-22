use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("polarhistogram"),
 slug: Some("polarhistogram"),
 summary: "Create histogram charts in polar coordinates.",
 description: concat!(
    "`polarhistogram(theta)` bins angular samples in radians and renders each bin as ",

    "a polar wedge whose radius is the bin count or normalized bin value. Theta ",

    "values wrap periodically into the selected angular range. The returned handle ",

    "reuses MATLAB-style histogram object properties such as `BinEdges`, ",

    "`BinCounts`, `Values`, `Data`, `NumBins`, `BinWidth`, `BinLimits`, ",

    "`Normalization`, `DisplayStyle`, `FaceColor`, `FaceAlpha`, `EdgeColor`, and ",

    "`DisplayName`.",

),

 keywords: &["polarhistogram",
 "polar histogram",
 "angular histogram",
 "theta bins",
 "matlab polarhistogram",
 "plotting"],
 related: &["histogram",
 "histcounts",
 "polarplot",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "All eight integer classes are accepted for angular data; native values gather ",

    "authoritatively before the explicit floating binning and rendering boundary, ",

    "while integer bin counts and table selectors remain structural.",

),
 concat!(
    "`polarhistogram(theta)` creates a histogram object handle and marks the target ",

    "axes as polar with equal x/y scaling. With no bin controls it uses six angular ",

    "bins spanning `[0, 2*pi]`.",

),
 concat!(
    "`polarhistogram(theta, nbins)` and `polarhistogram(theta, edges)` use the same ",

    "scalar bin-count and explicit edge-vector semantics as `histcounts`.",

),
 concat!(
    "Finite theta samples are wrapped into the chosen angular range before binning; ",

    "non-finite samples are ignored by the histogram counts and exposed as `NaN` in ",

    "normalized stored `Data`.",

),
 concat!(
    "Name/value bin controls such as `BinEdges`, `NumBins`, `BinWidth`, `BinLimits`, ",

    "`Normalization`, and `BinMethod` are forwarded through the same binning ",

    "implementation as `histogram`.",

),
 concat!(
    "`polarhistogram('BinEdges', edges, 'BinCounts', counts)` creates a polar ",

    "histogram from precomputed bin counts without sample data.",

),
 concat!(
    "`DisplayStyle` accepts `bar` and `stairs`; `stairs` renders the outer polar bin ",

    "arcs without filled radial wedges.",

),
 concat!(
    "Common bar styling arguments such as `FaceColor`, `EdgeColor`, `LineWidth`, ",

    "`BarWidth`, `FaceAlpha`, and `DisplayName` are accepted.",

),
 "The target axes form `polarhistogram(ax, ...)` plots into an existing axes handle.",
 concat!(
    "The returned handle has type `histogram`; `get` exposes histogram state and ",

    "supported `set` calls update `Normalization`, `DisplayName`, `DisplayStyle`, ",

    "`FaceColor`, `FaceAlpha`, and `EdgeColor`.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`polarhistogram` is a rendering sink. Angular samples are materialized for ",

    "`histcounts` binning, then the resulting bin counts are rendered as polar wedge ",

    "geometry through `runmat-plot`.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "polarhistogram declares no direct numeric provider precision list; host ",

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
    "Theta samples, explicit edges or named histogram controls use periodic radians. ",

    "Without explicit edges/range, the default range is 0..2*pi; without other bin ",

    "controls the actual default is six bins. Finite samples wrap to the selected ",

    "range. BinEdges/BinCounts without sample data accepts precomputed nonnegative ",

    "counts matching one fewer than strictly increasing edges. Supplying both ",

    "samples and manual counts rejects.",

),
 concat!(
    "Bar/stairs styles, normalization and supported shared bar properties produce a ",

    "histogram handle with polar metadata. Counts, normalized Values, Data and edges ",

    "are separate values. All real integer sample classes cross the explicit ",

    "floating normalization boundary; exact native integer polar properties are not ",

    "promised. GPU angular samples gather through their owner for host periodic ",

    "conversion; no direct provider polar-binning kernel or finite chart owner is ",

    "claimed.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Create a polar histogram",
 program: "theta = 2*pi*rand(1, 1000);\nh = polarhistogram(theta);",
 display_output: Some("% h is a histogram object handle"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==numel(theta)); assert(get(h,'NumBins')==6);",

) } },

BuiltinExample { id: "retained-2",
 title: "Use explicit angular bin edges",
 program: "theta = [0 0.2 1.0 2.0];\nedges = 0:pi/4:2*pi;\nh = polarhistogram(theta, edges);\nget(h, 'BinEdges');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(get(h,'BinEdges'),edges)); ",

    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==4); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Normalize and style the chart",
 program: concat!(
    "theta = 2*pi*rand(1, 500);\nh = polarhistogram(theta, 'NumBins', 16, ",

    "'Normalization', 'probability', 'FaceColor', 'red', 'DisplayName', ",

    "'angles');\nget(h, 'Normalization')",

),
 display_output: Some("ans =\n    'probability'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==500); assert(get(h,'NumBins')==16); ",

    "assert(strcmp(get(h,'Normalization'),'probability')); ",

    "assert(strcmp(get(h,'DisplayName'),'angles'));",

) } },

BuiltinExample { id: "retained-4",
 title: "Plot precomputed polar bin counts with stairs display",
 program: concat!(
    "edges = [0 pi 2*pi];\ncounts = [4 2];\nh = polarhistogram('BinEdges', edges, ",

    "'BinCounts', counts, 'DisplayStyle', 'stairs');\nget(h, 'BinCounts');",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(h,'BinCounts'); expected=counts; ",

    "assert(isequal(observed(:),expected(:))); ",

    "assert(strcmp(get(h,'DisplayStyle'),'stairs')); ",

    "assert(isequal(get(h,'BinEdges'),edges));",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Are theta values interpreted in degrees?",
 answer: "No. Like MATLAB, `polarhistogram` interprets theta samples and explicit bin edges in radians." },

BuiltinDocumentationFaq { question: "Does polarhistogram return a special handle type?",
 answer: concat!(
    "The returned object follows the histogram handle path, so `get(h, 'Type')` ",

    "returns `histogram` and histogram properties such as `BinEdges`, `BinCounts`, ",

    "`Values`, `Data`, `NumBins`, `BinWidth`, `BinLimits`, `Normalization`, ",

    "`DisplayStyle`, and `DisplayName` are available.",

) },

BuiltinDocumentationFaq { question: "What happens to angles outside 0 to 2*pi?",
 answer: concat!(
    "Finite values are wrapped periodically into the selected angular range before ",

    "binning. For example, `0`, `2*pi`, and `-2*pi` contribute to the same default ",

    "angular bin.",

) },

BuiltinDocumentationFaq { question: "Can polarhistogram plot GPU arrays?",
 answer: concat!(
    "GPU-resident inputs are accepted through the underlying histogram evaluation ",

    "path, but the current implementation gathers angular samples for binning before ",

    "rendering polar wedge geometry. A direct provider binning fast path belongs to ",

    "the queued GPU audit work.",

) }],

 links: &[BuiltinDocumentationLink { label: "histogram",
 target: BuiltinDocumentationLinkTarget::Builtin("histogram") },

BuiltinDocumentationLink { label: "histcounts",
 target: BuiltinDocumentationLinkTarget::Builtin("histcounts") },

BuiltinDocumentationLink { label: "polarplot",
 target: BuiltinDocumentationLinkTarget::Builtin("polarplot") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/polarhistogram.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/polarhistogram.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::polarhistogram::tests and runmat_plot::plots::bar::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "vm",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
