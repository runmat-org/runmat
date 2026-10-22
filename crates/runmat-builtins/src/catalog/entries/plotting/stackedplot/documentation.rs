use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("stackedplot"),
 slug: Some("stackedplot"),
 summary: "Create a stacked line chart with one y-axis per plotted variable.",
 description: concat!(
    "`stackedplot` creates a chart with vertically aligned axes that share one ",

    "x-axis. Pass a vector for one series, a matrix to place each column on a ",

    "separate axes, or paired `X,Y` data to choose the common x coordinates. Table ",

    "and timetable forms select plottable variables by name, numeric index, logical ",

    "mask, or supported selector object. Numeric source data remains in its native ",

    "`single`, `double`, logical, or integer storage for chart-property queries; ",

    "floating values used by the renderer are derived from that source rather than ",

    "becoming the chart's authoritative data.",

),

 keywords: &["stackedplot",
 "stacked plot",
 "stacked line chart",
 "table plot",
 "timetable plot",
 "matlab stackedplot"],
 related: &["plot",
 "table",
 "timetable",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "`Y` can be a vector or matrix. Row and column vectors become one stacked axes; ",

    "matrix columns become separate stacked axes.",

),
 "`X` must be a vector whose length matches the vector length or matrix row count of `Y`.",
 concat!(
    "Table and timetable inputs select plottable variables by default. Text, cell, ",

    "struct, and unsupported object variables are skipped unless explicitly ",

    "selected, in which case the chart errors if no plottable variables remain.",

),
 concat!(
    "`vars` and `DisplayVariables` use the shared table variable selector machinery ",

    "for table inputs, including names, string arrays, cellstr, numeric indices, ",

    "logical masks, and supported selector objects. `DisplayLabels` accepts text ",

    "scalars, string arrays, character arrays, or cell arrays of text scalars.",

),
 concat!(
    "Multiple table or multiple timetable inputs must be the same tabular kind. ",

    "Matching variable names combine on one stacked axes by default; ",

    "`CombineMatchingNames=false` separates them by input.",

),
 concat!(
    "`XVariable` selects the common x-axis variable from table inputs. It may be one ",

    "variable shared by all table inputs or one variable per input table. Timetable ",

    "row times are used automatically, and `XVariable` is rejected for timetables.",

),
 "LineSpec and line style name-value properties apply to every child line.",
 concat!(
    "The returned handle is a `stackedplot` chart object. `get` supports `Type`, ",

    "`Parent`, `DisplayVariables`, `SourceTable`, `XVariable`, ",

    "`CombineMatchingNames`, `XData`, `YData`, `XLabel`, `Title`, `Visible`, ",

    "`GridVisible`, `XLimits`, `LineWidth`, `LineStyle`, and `Color`. `YData` is a ",

    "numeric matrix when child lines have matching lengths and a cell array of ",

    "vectors when multiple source tables have unequal lengths.",

),
 concat!(
    "`SourceTable` returns a lightweight provenance snapshot with source table ",

    "classes and variable names rather than embedding the original table objects in ",

    "plotting registry state.",

),
 concat!(
    "All eight integer classes are accepted for numeric X/Y data and for numeric ",

    "table-variable selectors. Integer selectors are decoded as exact positive ",

    "indices, so adjacent `uint64` values are never collapsed through binary64 ",

    "conversion.",

),
 concat!(
    "`XData` and `YData` preserve native numeric storage and exact integer values. ",

    "When all Y columns share a class and length, `YData` is rebuilt as one ",

    "same-class matrix; heterogeneous classes or unequal lengths are returned as a ",

    "cell array of typed vectors.",

),
 concat!(
    "`set` supports `Title`, `XLabel`, `DisplayLabels`, `YLabels`, `Visible`, ",

    "`GridVisible`, `XLimits`, `Color`, `LineStyle`, `LineWidth`, `Marker`, and ",

    "`MarkerSize`. `XLimits` accepts finite or infinite numeric bounds as long as ",

    "the range is increasing.",

),
 concat!(
    "Figure handles are accepted as parents. Axes handles, nested grouped ",

    "`DisplayVariables` cells, event-table rendering, interactive chart controls, ",

    "and native provider-side column splitting are not implemented yet. ",

    "Automatically resident inputs can use an exact owner-aware host fallback, while ",

    "explicit GPU input requires RunMat mode.",

)] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "`stackedplot` is a composite host chart rather than a numeric GPU kernel. ",

    "Automatically resident data is gathered through its owning provider before ",

    "columns are split across stacked axes. An explicitly supplied `gpuArray` is a ",

    "separately gated RunMat extension because the public chart surface does not ",

    "document direct interactive GPU input.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "stackedplot declares no direct numeric provider precision list; host ",

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
    "Numeric Y, X/Y, table/timetable, selected variables, multiple same-kind tabular ",

    "inputs and supported cell-of-tabular inputs retain the existing shared table ",

    "selector machinery. Figure parents are supported; axes parents reject. Native ",

    "source arrays and selected native columns remain authoritative while floating ",

    "line geometry is derived once per selected column. Vectors make one axes; ",

    "matrix columns make stacked axes; the actual maximum is 25 variables.",

),
 concat!(
    "XVariable/CombineMatchingNames/DisplayVariables/DisplayLabels/YLabels/Title/XLab",

    "el/Visible/GridVisible/XLimits and shared line style controls retain their real ",

    "grammar. Timetables use row times and reject XVariable; ",

    "numeric/datetime/duration/categorical variable conversions remain existing ",

    "domain owners. SourceTable is lightweight provenance rather than retained ",

    "original tables. Grouped selector cells, event rendering and interactive ",

    "controls remain explicit gaps. Unsupported tabular variables are skipped by the ",

    "established selection behavior; no fake data columns are substituted.",

),
 concat!(
    "Automatic resident numeric values use the existing owning-provider readback ",

    "before host split. Explicit GPU inputs retain their separate compatibility ",

    "gate, which is not by itself placement consent or proof of finite readback ",

    "admission. This migration does not certify unimplemented explicit placement ",

    "routes, finite table-property publication or physical provider/renderer ",

    "lifetimes.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Plot each matrix column in a separate stacked axes",
 program: "Y = [1 10; 2 20; 3 30];\ns = stackedplot(Y);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(strcmp(get(s,'Type'),'stackedplot')); observed=get(s,'YData'); ",

    "expected=Y; assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use a common x vector and line style",
 program: "x = [0 1 2];\nY = [1 10; 2 20; 3 30];\nstackedplot(x, Y, '--o', 'LineWidth', 1.5);",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gcf,'Type','stackedplot'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'YData'); expected=Y; ",

    "assert(isequal(observed(:),expected(:))); observed=get(witness(1),'XData'); ",

    "expected=x; assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Plot selected table variables",
 program: concat!(
    "T = table([1;2;3], [4;5;6], [7;8;9], 'VariableNames', {'Time','A','B'});\ns = ",

    "stackedplot(T, {'A','B'}, 'XVariable', 'Time');\nget(s, 'DisplayVariables')",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(s,'YData'); expected=[4 7;5 8;6 9]; ",

    "assert(isequal(observed(:),expected(:))); labels=get(s,'DisplayVariables'); ",

    "assert(numel(labels)==2); assert(strcmp(labels(1),'A')&&strcmp(labels(2),'B'));",

) } },

BuiltinExample { id: "retained-4",
 title: "Combine matching variables from multiple tables",
 program: concat!(
    "T1 = table([1;2], 'VariableNames', {'A'});\nT2 = table([3;4], 'VariableNames', ",

    "{'A'});\ns = stackedplot(T1, T2, 'A');",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: "assert(strcmp(get(s,'Type'),'stackedplot')); observed=get(s,'YData'); assert(numel(observed)==4);" } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Does stackedplot return line handles?",
 answer: concat!(
    "No. RunMat returns a first-class `stackedplot` chart handle that owns the child ",

    "line plots and stacked axes.",

) },

BuiltinDocumentationFaq { question: "Does stackedplot preserve gpuArray residency?",
 answer: concat!(
    "Not currently. Matrix and table inputs must be split into stacked axes and ",

    "grouped child lines, so RunMat materializes inputs before handing each child ",

    "line to the renderer. This is tracked for the dedicated GPU/provider audit ",

    "rather than hidden behind a partial fast path.",

) }],

 links: &[BuiltinDocumentationLink { label: "plot",
 target: BuiltinDocumentationLinkTarget::Builtin("plot") },

BuiltinDocumentationLink { label: "table",
 target: BuiltinDocumentationLinkTarget::Builtin("table") },

BuiltinDocumentationLink { label: "timetable",
 target: BuiltinDocumentationLinkTarget::Builtin("timetable") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/stackedplot.rs`",
 target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",

    "s/plotting/ops/stackedplot.rs",

)) }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::stackedplot::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "vm",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
