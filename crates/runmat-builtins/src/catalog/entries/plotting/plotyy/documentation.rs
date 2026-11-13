use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("plotyy"),
 slug: Some("plotyy"),
 summary: "Plot two data sets with independent left and right Y axes.",
 description: concat!(
    "`plotyy` creates two overlaid Cartesian axes in the current figure, plots the ",

    "first data set against the left Y axis, plots the second data set against the ",

    "right Y axis, and returns the axes and line handles. It is a legacy MATLAB ",

    "compatibility helper; new MATLAB code generally uses `yyaxis`.",

),

 keywords: &["plotyy",
 "dual y axis",
 "plotting",
 "legacy"],
 related: &["plot",
 "line",
 "semilogx",
 "semilogy",
 "loglog"],

 sections: &[BuiltinDocumentationSection { heading: "Usage notes",
 paragraphs: &["`plotyy` returns the left axes as the current axes after plotting.",
 "The right axes shares the primary plot viewport and stores `YAxisLocation` as `right`.",
 concat!(
    "For compatibility with existing RunMat plotting primitives, line creation uses ",

    "the same handle and data-shape behavior as `line`.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`plotyy` is a plotting sink. It delegates line creation to the existing ",

    "low-level `line` path, so GPU-resident inputs follow the same residency ",

    "behavior as `line` and are not gathered by `plotyy` itself.",

)] },

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
    "Four coordinate operands and zero, one or two plotting selectors create ",

    "left/right axes; the single selector applies to both. Selectors are the ",

    "implemented plot/line/semilogx/semilogy/loglog names or corresponding handle ",

    "forms, not arbitrary callback execution. No general user-function invocation, ",

    "Unknown user-code effect or Process-only stack requirement is invented. Zero ",

    "through three outputs select ax=[left right], h1 and h2; the left axes becomes ",

    "current.",

),
 concat!(
    "The actual context roots each derived coordinate pair through the existing ",

    "borrowed line parser. Source class/shape and multi-series expansion follow that ",

    "owner, including its native property selector and installed renderer route. ",

    "Dual-axis display modes are set explicitly. Provider, cancellation and ",

    "line-data failures preserve typed causes and terminal replay policy; ",

    "unsupported selector names reject before axes creation.",

)] },

BuiltinDocumentationSection { heading: "Retained call forms",
 paragraphs: &["ax = plotyy(X1, Y1, X2, Y2)",
 "[ax, h1, h2] = plotyy(X1, Y1, X2, Y2)",
 "[ax, h1, h2] = plotyy(X1, Y1, X2, Y2, fun)",
 "[ax, h1, h2] = plotyy(X1, Y1, X2, Y2, fun1, fun2)"] },

BuiltinDocumentationSection { heading: "Retained argument details",
 paragraphs: &["X1, Y1",
 "numeric arrays",
 concat!(
    "Data plotted against the left Y axis. Data shape handling is delegated to the ",

    "low-level `line` primitive.",

),
 "X2, Y2",
 "numeric arrays",
 concat!(
    "Data plotted against the right Y axis. Data shape handling is delegated to the ",

    "low-level `line` primitive.",

),
 "fun, fun1, fun2",
 "function handle | string | character vector",
 concat!(
    "Optional plotting function selectors. Supported values are `plot`, `line`, ",

    "`semilogx`, `semilogy`, and `loglog`.",

)] },

BuiltinDocumentationSection { heading: "Retained output details",
 paragraphs: &["ax",
 "numeric row vector",
 "Two axes handles `[left right]`.",
 "h1",
 "numeric scalar | numeric row vector",
 "Line handle or handles for the left axes.",
 "h2",
 "numeric scalar | numeric row vector",
 "Line handle or handles for the right axes."] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Plot two series with separate Y axes",
 program: "[ax,h1,h2] = plotyy(1:3, [10 20 30], 1:3, [100 400 900]);\nget(ax(2), 'YAxisLocation')",
 display_output: Some("right"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(size(ax),[1 2])); assert(gca==ax(1)); ",

    "assert(strcmp(get(ax(2),'YAxisLocation'),'right')); observed=get(h1,'YData'); ",

    "expected=[10 20 30]; assert(isequal(observed(:),expected(:))); ",

    "observed=get(h2,'YData'); expected=[100 400 900]; ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use logarithmic plotting functions",
 program: concat!(
    "[ax,h1,h2] = plotyy(1:3, [10 20 30], 1:3, [100 400 900], 'semilogx', ",

    "'semilogy');\n{get(ax(1),'XScale'), get(ax(2),'YScale')}",

),
 display_output: Some("{'log', 'log'}"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(ax(1),'XScale'),'log')); ",

    "assert(strcmp(get(ax(2),'YScale'),'log')); assert(gca==ax(1)); ",

    "observed=get(h2,'YData'); expected=[100 400 900]; ",

    "assert(isequal(observed(:),expected(:))); ",

) } }],

 example_exemption: None,

 faqs: &[],

 links: &[BuiltinDocumentationLink { label: "plot",
 target: BuiltinDocumentationLinkTarget::Builtin("plot") },

BuiltinDocumentationLink { label: "line",
 target: BuiltinDocumentationLinkTarget::Builtin("line") },

BuiltinDocumentationLink { label: "semilogx",
 target: BuiltinDocumentationLinkTarget::Builtin("semilogx") },

BuiltinDocumentationLink { label: "semilogy",
 target: BuiltinDocumentationLinkTarget::Builtin("semilogy") },

BuiltinDocumentationLink { label: "loglog",
 target: BuiltinDocumentationLinkTarget::Builtin("loglog") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/plotting/ops/plotyy.rs",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/plotyy.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::plotyy::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "vm",
 location: "runmat-vm plotting::plotyy_dispatches_and_returns_dual_axes_outputs" }],
 notes: &[] },

 introduced: Some("0.5.3"),
 status: Some(BuiltinDocumentationStatus::Partial),

};
