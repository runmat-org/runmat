use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("hist"),
 slug: Some("hist"),
 summary: "Create legacy center-based histograms in MATLAB and RunMat.",
 description: concat!(
    "`hist` creates legacy histograms using bin-center semantics. When the second ",

    "argument is a vector, it is interpreted as bin centers rather than bin edges, ",

    "consistent with MATLAB behavior.",

),

 keywords: &["hist",
 "legacy histogram",
 "bin centers",
 "distribution plot",
 "matlab hist",
 "hist vs histogram"],
 related: &["histogram",
 "bar",
 "pie"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`hist(data)` uses ten bins by default. Ordinary row and column vectors create ",

    "one distribution; matrix columns use the existing per-column bin selection.",

),
 concat!(
    "`hist(data, v)` interprets `v` as bin centers, not bin edges. This is the key ",

    "semantic difference from `histogram`.",

),
 concat!(
    "`hist` is a plotting-style histogram command, while `histogram` returns a ",

    "first-class histogram object handle.",

),
 concat!(
    "Normalization and bin-control workflows are supported, but the builtin should ",

    "still be documented and used as the legacy center-based histogram path rather ",

    "than as an alias of `histogram`.",

),
 concat!(
    "GPU-aware computation is used where supported, but semantics stay aligned with ",

    "legacy MATLAB `hist` behavior.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`hist` supports GPU-aware histogram computation where applicable, but it ",

    "remains the legacy center-based histogram workflow rather than the object-based ",

    "`histogram` API.",

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
    "The legacy API returns zero, one or two requested outputs: double counts and ",

    "optionally centres, while retaining the existing plotting behavior. The default ",

    "is ten bins, not a sample-size formula. A positional vector supplies centres, a ",

    "scalar supplies a validated count. Integer sample/centre and modern option ",

    "forms retain their distinct existing compatibility extensions; integer count ",

    "controls are structural. Modern BinEdges/NumBins/BinWidth/BinLimits/BinMethod/We",

    "ights/Normalization controls retain their existing exclusivity and validation ",

    "rules.",

),
 concat!(
    "Ordinary 1-by-N and N-by-1 vectors now use one distribution and return row ",

    "counts/centres. Matrix columns still form separate grouped distributions; ",

    "automatic/count bins are selected per column, while explicit/shared controls ",

    "retain the old common-bin owner. No new N-D or empty geometry rule is ",

    "introduced by that bounded vector correction. Probability and PDF weights use ",

    "the existing per-column totals and widths.",

),
 concat!(
    "The eligible resident floating-vector path preserves the existing typed GPU ",

    "histogram preparation states: complete chart, prepared host bins, or required ",

    "host statistics. Continuing already successful preparation is not retry ",

    "permission: after-effect failures remain terminal. Unsupported styling or ",

    "pre-effect inapplicability retains the existing owner-aware CPU route, not a ",

    "provider-name heuristic. Finite workspace/renderer completion and CPU long-loop ",

    "cooperation remain unclosed. Bar getters do not expose count vectors; examples ",

    "separately assert the real returned numerical counts and retained Runtime ",

    "fixtures inspect chart data.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Create a default legacy histogram",
 program: "data = randn(1, 1000);\nhist(data);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','bar'); assert(numel(witness)>=1); ",

    "assert(numel(witness)==1); [observed,observedCentres]=hist(data); ",

    "assert(sum(observed(:))==numel(data)); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Pass explicit bin centers",
 program: "data = randn(1, 500);\ncenters = -3:0.5:3;\nhist(data, centers);",
 display_output: Some("% Here the second argument is interpreted as bin centers, not edges"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','bar'); assert(numel(witness)>=1); ",

    "assert(numel(witness)==1); [observed,observedCentres]=hist(data,centers); ",

    "assert(sum(observed(:))==numel(data)); assert(numel(observed)==numel(centers));",

) } },

BuiltinExample { id: "retained-3",
 title: "Compare legacy hist semantics with modern histogram intent",
 program: "data = randn(1, 300);\nhist(data, -2:0.5:2);",
 display_output: Some("% Use histogram(...) instead when you want bin-edge semantics and an object handle"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','bar'); assert(numel(witness)>=1); ",

    "assert(numel(witness)==1); [observed,observedCentres]=hist(data,-2:0.5:2); ",

    "assert(sum(observed(:))==numel(data)); assert(numel(observed)==9);",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "What's the actual difference between hist and histogram?",
 answer: concat!(
    "The critical difference is how the second argument is interpreted. `hist(data, ",

    "v)` treats `v` as bin centres; `histogram(data, 'BinEdges', v)` treats `v` as ",

    "bin edges. `histogram` returns a chart handle with get/set properties. `hist` ",

    "is the legacy plotting/count API: one requested output returns double counts ",

    "and two return counts and centres; it is not a histogram object alias.",

) },

BuiltinDocumentationFaq { question: "When should I use hist instead of histogram?",
 answer: concat!(
    "Use `hist` when you're porting legacy MATLAB code that relies on center-based ",

    "binning or when you want a quick distribution plot without needing a handle ",

    "object. For anything new—especially if you need normalization modes, bin-edge ",

    "control, or programmatic access to bin counts—use `histogram` instead.",

) }],

 links: &[BuiltinDocumentationLink { label: "histogram",
 target: BuiltinDocumentationLinkTarget::Builtin("histogram") },

BuiltinDocumentationLink { label: "bar",
 target: BuiltinDocumentationLinkTarget::Builtin("bar") },

BuiltinDocumentationLink { label: "pie",
 target: BuiltinDocumentationLinkTarget::Builtin("pie") },

BuiltinDocumentationLink { label: "Choosing the right plot type",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },

BuiltinDocumentationLink { label: "Styling plots and axes",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },

BuiltinDocumentationLink { label: "Plot replay and export",
 target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },

BuiltinDocumentationLink { label: "Complete plotting guide",
 target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/hist.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/hist.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::hist::tests" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
