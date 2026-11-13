use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("plotmatrix"),
 slug: Some("plotmatrix"),
 summary: "Create a scatter plot matrix from matrix columns.",
 description: concat!(
    "`plotmatrix` builds a grid of axes where each off-diagonal subplot compares one ",

    "variable column against another. The one-input form also places histograms on ",

    "the diagonal. Matrix rows are observations and columns are variables; vector ",

    "inputs are treated as one variable.",

),

 keywords: &["plotmatrix",
 "scatter",
 "matrix plot",
 "plotting",
 "statistics"],
 related: &["scatter",
 "histogram",
 "subplot"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`plotmatrix(X)` creates an `n` by `n` grid for the `n` variables in `X`, with ",

    "diagonal histograms and off-diagonal scatter plots.",

),
 concat!(
    "`plotmatrix(X,LineSpec)` applies the line specification to off-diagonal scatter ",

    "plots while retaining diagonal histograms.",

),
 concat!(
    "`plotmatrix(X,Y)` creates an `n` by `m` rectangular grid for `n` columns in `X` ",

    "and `m` columns in `Y`; diagonal histogram outputs are empty for this form.",

),
 "`plotmatrix(X,Y,lineSpec)` applies the line specification to scatter plots in the matrix.",
 "`plotmatrix(ax,...)` uses the supplied axes as the compatibility big-axes target.",
 concat!(
    "`[S,AX,BigAx,H,HAx] = plotmatrix(X)` returns the scatter handle matrix, axes ",

    "handle matrix, current big-axes handle, diagonal histogram handles, and ",

    "diagonal histogram axes handles.",

),
 concat!(
    "Inputs must be numeric and must have the same number of observations when both ",

    "`X` and `Y` are supplied.",

),
 "Diagonal histograms are created through the same `histogram` builtin path as direct histogram calls."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`plotmatrix` is a rendering sink. MATLAB accepts gpuArray inputs for this ",

    "builtin but executes on the CPU. RunMat gathers input matrices when needed, ",

    "slices columns on the host, and delegates rendering to existing `scatter` and ",

    "`histogram` plotting paths.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "plotmatrix declares no direct numeric provider precision list; host ",

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
    "One matrix, optional LineSpec, or paired matrices with optional LineSpec ",

    "accepts an optional full leading axes target. Inputs must be nonempty numeric ",

    "data; vectors are one variable and matrix columns variables, rows observations. ",

    "Paired matrices require equal observation counts. S and AX are variable-count ",

    "matrices; BigAx is scalar; one-input H and HAx are diagonal histogram handle ",

    "rows, and the paired form has empty diagonal outputs. Maximum output count is ",

    "five.",

),
 concat!(
    "The actual caller context and rooted original operands survive gathering and ",

    "every delegated scatter/histogram await. Histogram child argument slots use ",

    "canonical context-aware ingress; selected native columns use the existing ",

    "borrowed selector rather than copying the full input per column. Direct integer ",

    "source properties remain native at child scatter boundaries. BigAx remains the ",

    "compatibility axes implementation, not a fabricated fully equivalent MATLAB ",

    "invisible spanning-axes model.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Plot pairwise relationships between columns",
 program: "X = [1 10; 2 20; 3 30; 4 40];\nS = plotmatrix(X);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(size(S),[2 2])); assert(isnan(S(1,1))&&isnan(S(2,2))); ",

    "assert(strcmp(get(S(2,1),'Type'),'scatter')); observed=get(S(2,1),'XData'); ",

    "expected=X(:,2); assert(isequal(observed(:),expected(:))); ",

    "observed=get(S(2,1),'YData'); expected=X(:,1); ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Compare X columns against a separate response column",
 program: "X = [1 10; 2 20; 3 30];\nY = [4; 5; 6];\n[S,AX] = plotmatrix(X, Y, 'r+');",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(size(S),[2 1])); assert(isequal(size(AX),[2 1])); ",

    "observed=get(S(1),'YData'); expected=Y; assert(isequal(observed(:),expected(:)))",

    "; observed=get(S(2),'XData'); expected=X(:,2); ",

    "assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Inspect diagonal histogram handles",
 program: "X = [1 10; 2 20; 3 30];\n[S,AX,BigAx,H,HAx] = plotmatrix(X);\nnumel(H)",
 display_output: Some("ans =\n     2"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(numel(H)==2&&numel(HAx)==2); assert(gca==BigAx); ",

    "assert(strcmp(get(H(1),'Type'),'histogram')); counts=get(H(1),'BinCounts'); ",

    "assert(sum(counts(:))==3); assert(strcmp(get(H(2),'Type'),'histogram')); ",

    "counts=get(H(2),'BinCounts'); assert(sum(counts(:))==3); ",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Does plotmatrix preserve gpuArray residency?",
 answer: concat!(
    "No. `plotmatrix` is a CPU-rendered plotting sink, matching MATLAB's documented ",

    "gpuArray behavior for this function. Individual scatter/histogram rendering ",

    "then follows the existing plotting paths.",

) },

BuiltinDocumentationFaq { question: "What is BigAx in RunMat?",
 answer: concat!(
    "RunMat returns a valid axes handle and leaves it as `gca` so scripts can pass ",

    "it to labeling/property helpers. The current renderer does not model MATLAB's ",

    "invisible spanning axes as a distinct object.",

) }],

 links: &[BuiltinDocumentationLink { label: "scatter",
 target: BuiltinDocumentationLinkTarget::Builtin("scatter") },

BuiltinDocumentationLink { label: "histogram",
 target: BuiltinDocumentationLinkTarget::Builtin("histogram") },

BuiltinDocumentationLink { label: "subplot",
 target: BuiltinDocumentationLinkTarget::Builtin("subplot") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/plotmatrix.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/plotmatrix.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::plotmatrix::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "integration",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
