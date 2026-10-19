use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("grpstats"),
    slug: Some("grpstats"),
    summary: "Compute summary statistics organized by group.",
    description: "`grpstats(X,group)` computes grouped statistics for numeric or logical matrices. `grpstats(T,groupvars,...)` returns a table containing grouping variables, group counts, and requested summary columns.",
    keywords: &["grpstats", "group", "statistics", "mean", "std", "confidence interval", "table"],
    related: &[],
    sections: &[
        BuiltinDocumentationSection { heading: "Matrix and group input", paragraphs: &[
            "Matrix grouping variables may be numeric, logical, text, categorical, or a cell array of grouping vectors. Empty grouping computes one all-row group. Missing group labels are ignored and NaN data are omitted from numeric summaries.",
            "Supported statistics are mean, sem, std, var, min, max, range, meanci, predci, numel, and gname. Multiple requested statistics return separate outputs, and the requested output count must match the selected statistics.",
            "`meanci` and `predci` return `ngroups`-by-`ncols`-by-2 interval tensors, honor the `Alpha` option, and use Student's t critical values.",
        ] },
        BuiltinDocumentationSection { heading: "Table input", paragraphs: &[
            "Grouping variables use table selectors; `[]` selects no grouping. Output includes grouping variables, `GroupCount`, and one column per selected data variable/statistic. Interval statistics occupy two-column table variables.",
            "`DataVars` selects numeric or logical columns and `VarNames` replaces output variable names. Built-in handles such as `@mean` and `@std` map to supported statistics. Custom function handles are rejected until general callback dispatch is implemented for this function.",
            "Integer grouping keys are compared exactly. Integer matrix data are a RunMat-only form and cross a checked double computation boundary; integer table extrema retain their original class, while floating statistics convert only at the calculation boundary.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Grouping runs on the host after any required gather. It is a fusion boundary, not a GPU reduction. Explicit interactive resident input and native integer matrix data are independently mode-gated RunMat extensions.",
            "Static analysis can distinguish a table result from a matrix result and type literal-named matrix statistics. Group count and table columns depend on runtime data and are not inferred from the table class alone.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "matrix-means", title: "Compute group means for a matrix", program: "X = [1 10; 3 30; 2 20; 4 40]; g = [2;1;2;1]; m = grpstats(X,g);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(m,[3.5 35;1.5 15]));" } },
        BuiltinExample { id: "multiple-statistics", title: "Return multiple statistics and group names", program: "X = [1 10; 3 30; 2 20; 4 40]; g = [2;1;2;1]; [m,s,names] = grpstats(X,g,[\"mean\",\"std\",\"gname\"]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(m,[3.5 35;1.5 15])); assert(isequal(size(s),[2 2])); assert(iscell(names)); assert(numel(names)==2);" } },
        BuiltinExample { id: "selected-table-data", title: "Summarize selected table variables", program: "T = table([2;1;2;1],[1;3;2;4],[10;30;20;40],'VariableNames',{'G','X','Y'}); S = grpstats(T,'G',[\"mean\",\"max\"],'DataVars','X');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(S)); assert(width(S)==4); assert(isequal(S.G,[1;2])); assert(isequal(S.GroupCount,[2;2])); assert(isequal(S.mean_X,[3.5;1.5])); assert(isequal(S.max_X,[4;2]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does grpstats support function handles?", answer: "Built-in statistic handles such as @mean and @std work. Custom handles are not implemented for grpstats." },
        BuiltinDocumentationFaq { question: "How are groups ordered?", answer: "Numeric and logical groups sort by value. Text groups retain first-seen order." },
    ],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/object/analytics/grpstats.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Matrix and table grouped-statistics tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM grouped statistics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &["Legacy behavior, FAQs, source, and examples are retained; every example now carries standalone setup and assertions.", "Legacy gpu_support and fusion map to typed placement. requires_feature was null and tested maps to verification evidence."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
