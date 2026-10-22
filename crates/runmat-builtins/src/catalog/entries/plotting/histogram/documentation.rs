use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("histogram"),
 slug: Some("histogram"),
 summary: "Create histogram chart objects in MATLAB and RunMat.",
 description: concat!(
    "`histogram` creates object-style histograms using bin-edge semantics. ",

    "Normalization options, bin controls, and returned handle properties follow ",

    "MATLAB behavior.",

),

 keywords: &["histogram",
 "histogram object",
 "bin edges",
 "normalized histogram",
 "matlab histogram",
 "distribution analysis"],
 related: &["hist",
 "bar",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &["`histogram(data)` creates a histogram object handle.",
 concat!(
    "When you pass explicit bins through histogram-style arguments, they are treated ",

    "as bin edges rather than bin centers.",

),
 concat!(
    "Normalization modes such as count, probability, percentage, density-style ",

    "variants, and cumulative forms are part of the evaluation path.",

),
 concat!(
    "The returned histogram handle exposes properties like bin edges, counts, and ",

    "normalization through `get` and accepts updates through `set` where supported.",

),
 concat!(
    "This is the preferred histogram API for new code; use `hist` only when you ",

    "intentionally want legacy center-based MATLAB semantics.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &["`histogram` uses histogram evaluation with a handle-based plotting object interface."] },

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
    "The object API accepts sample data, optional count/edge controls, optional full ",

    "leading axes and bar styling. Data and exact integer edges retain the existing ",

    "histcounts evaluation owner; unlike hist, a positional edge vector means edges, ",

    "not centres. Raw double counts are evaluated once, then normalized for ",

    "rendering and stored with typed Data, edges, denominator and style metadata on ",

    "the histogram handle.",

),
 concat!(
    "BinEdges/NumBins/BinWidth/BinLimits/BinMethod and ",

    "count/probability/percentage/density/PDF/cumulative normalization retain the ",

    "actual underlying validation and supported get/set surface. Shape admission, ",

    "nonfinite handling and integer membership comparisons belong to histcounts. The ",

    "full figure+axes identity is selected before publication rather than reusing a ",

    "foreign axes index in the active figure. Resident Data is eagerly retained by ",

    "owning-provider readback before evaluation; this is not a direct no-readback ",

    "histogram-object kernel.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Create a histogram object from data",
 program: "data = randn(1, 1000);\nh = histogram(data);",
 display_output: Some("% h is a histogram object handle"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==numel(data)); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use explicit bin edges and inspect the handle",
 program: "data = randn(1, 500);\nedges = -3:0.25:3;\nh = histogram(data, 'BinEdges', edges);\nget(h, 'BinEdges');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(get(h,'BinEdges'),edges)); ",

    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==sum(data>=edges(1)&data<=edges(end))); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Apply normalization through histogram semantics",
 program: "data = randn(1, 500);\nh = histogram(data, 'Normalization', 'probability');\nget(h, 'Normalization')",
 display_output: Some("ans =\n    'probability'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(h,'Type'),'histogram')); counts=get(h,'BinCounts'); ",

    "assert(sum(counts(:))==numel(data)); assert(strcmp(get(h,'Normalization'),'proba",

    "bility')); values=get(h,'Values'); assert(abs(sum(values(:))-1)<1e-12);",

) } },

BuiltinExample { id: "retained-4",
 title: "Overlaid distributions",
 program: concat!(
    "a = 2 + 1.2*randn(1, 5000);\nb = 4 + 0.8*randn(1, 5000);\n\nh1 = histogram(a, ",

    "'BinEdges', -2:0.25:8, 'Normalization', 'probability');\nset(h1, 'DisplayName', ",

    "'Process A');\nhold on;\nh2 = histogram(b, 'BinEdges', -2:0.25:8, ",

    "'Normalization', 'probability');\nset(h2, 'DisplayName', 'Process B');\nhold ",

    "off;\n\ntitle('Distribution Comparison');\nxlabel('Measurement');\nylabel('Probabili",

    "ty');\nlegend;\ngrid on;",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(h1,'DisplayName'),'Process A')); ",

    "assert(strcmp(get(h2,'DisplayName'),'Process B')); ",

    "assert(strcmp(get(h1,'Normalization'),'probability')); ",

    "assert(strcmp(get(h2,'Normalization'),'probability')); ",

    "assert(numel(findobj(gca,'Type','histogram'))==2);",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "How do I set a specific number of bins?",
 answer: concat!(
    "Pass the bin count as a second positional argument or use the `'NumBins'` ",

    "name-value pair.\n\n```matlab\ndata = randn(1, 1000);\nhistogram(data, 30);         ",

    "           % 30 bins\nhistogram(data, 'NumBins', 50);         % 50 ",

    "bins\nhistogram(data, 'BinEdges', -3:0.1:3); % exact edges\n```\n\nExplicit ",

    "`BinEdges` gives you full control over where each bin starts and ends.",

) },

BuiltinDocumentationFaq { question: "How do I overlay two distributions on the same axes?",
 answer: concat!(
    "Use `hold on` between `histogram` calls and set `'Normalization'` to ",

    "`'probability'` so both distributions are on a comparable ",

    "scale.\n\n```matlab\nhistogram(randn(1,1000), 'Normalization', ",

    "'probability');\nhold on;\nhistogram(randn(1,1000) + 2, 'Normalization', ",

    "'probability');\nlegend('dist A', 'dist B');\n```",

) },

BuiltinDocumentationFaq { question: "What's the difference between histogram and hist?",
 answer: concat!(
    "`histogram` is the modern API: it uses bin-edge semantics, returns a handle ",

    "object you can inspect with `get`/`set`, and supports normalization modes like ",

    "`'probability'` and `'pdf'`. `hist` is the legacy API that interprets a second ",

    "vector argument as bin centers. Use `histogram` for new code unless you ",

    "specifically need center-based binning from a legacy workflow.",

) }],

 links: &[BuiltinDocumentationLink { label: "hist",
 target: BuiltinDocumentationLinkTarget::Builtin("hist") },

BuiltinDocumentationLink { label: "bar",
 target: BuiltinDocumentationLinkTarget::Builtin("bar") },

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

 media: &[BuiltinDocumentationMedia { kind: BuiltinDocumentationMediaKind::Image,
 url: "https://web.runmatstatic.com/builtin-image/runmat-matlab-histogram-plot-overlaid-distributions.webp",
 alt: "Overlaid distributions" }],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/histogram.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/histogram.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::histogram::tests" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
