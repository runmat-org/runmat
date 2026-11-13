use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("anova1"),slug:Some("anova1"),
    summary:"One-way analysis of variance for grouped observations.",
    description:"Compare two or more group means with the one-way F-test, returning an ANOVA table and group statistics.",
    keywords:&["anova1","ANOVA","F test","group means","analysis of variance"],related:&["boxplot","ttest2","findgroups"],
    sections:&[
        BuiltinDocumentationSection{heading:"Observation grouping",paragraphs:&[
            "anova1(y) treats the columns of a real single or double matrix as groups. anova1(y,group) accepts one grouping label per matrix column; repeated labels combine those columns. For vector observations group is required and supplies one label per observation. [] selects default column groups for a matrix.",
            "Grouping accepts a real single/double numeric vector, logical vector, character rows, string array, cell array of text scalars, or categorical observations. Labels retain typed numeric identity; they are not compared by rounded display strings. Groups are returned in first-observed order. Empty/missing labels and NaN labels omit corresponding observations; NaN y values are omitted independently. All-empty groups are removed. At least two observed groups and positive within-group degrees of freedom are required.",
            "Native integer, logical, complex, object, and distributed sample arrays are unsupported. Native integer grouping arrays are also unsupported. Numeric computation and outputs use host binary64, including single-input observations; no single-output-class parity is claimed.",
        ]},
        BuiltinDocumentationSection{heading:"Outputs and display",paragraphs:&[
            "[p,tbl,stats] returns the upper-tail F-test probability; a 4-by-6 cell table with Source, SS, df, MS, F, and Prob>F columns; and a scalar statistics structure. The table rows are header, Groups, Error, and Total. Unused cells are empty character rows. stats contains gnames (column cellstr), n and means (row vectors), source ('anova1'), error degrees of freedom df, and pooled within-group standard deviation s.",
            "The default display is 'on': RunMat prints the ANOVA table in the console and renders the groups through its boxplot implementation with notches. This is not MATLAB's separate interactive table window. The plot-core feature is required for 'on'; a build without plotting rejects that form explicitly. anova1(y,group,'off') suppresses both presentation actions and returns only the requested outputs. Unknown display values and excess inputs or outputs raise errors.",
            "A zero within-group variance with distinct group means gives an infinite F statistic and p=0. If every observation is identical the F ratio and p are undefined (NaN). Infinite observations can also make statistics undefined. This function supplies group summaries; it does not implement multcompare or claim the complete statistics-toolbox surface.",
        ]},
        BuiltinDocumentationSection{heading:"Execution and resource limits",paragraphs:&[
            "The host implementation reuses RunMat's regularized-beta probability kernel for F survival. Only authenticated automatic residency may use the context-owned host gather fallback, for observations and grouping values alike. Explicit gpuArray or unproven device intent is rejected before any argument transfer; call gather explicitly to request host analysis. There is no provider-native ANOVA kernel, numerical fusion producer, distributed implementation, or separately enabled explicit-host-fallback extension; original gather and cancellation failures propagate.",
            "Observation scans and moments cooperate with the owning execution context. Finite host-memory requests fail before gather or group/result allocation: those workspaces have no complete allocation certificate. Display='off' has no console, host-callback, or UI effects; unknown/default display retains the conservative presentation effects.",
        ]},
    ],
    examples:&[
        BuiltinExample{id:"matrix-groups",title:"Compare matrix columns",program:"y = [1 4;2 5;3 6]; [p,tbl,s] = anova1(y,{'A','B'},'off');",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(tbl{2,3}==1 && tbl{3,3}==4); assert(abs(tbl{2,5}-13.5)<1e-10); assert(p<0.05); assert(isequal(s.n,[3 3])); assert(isequal(s.means,[2 5]));"}},
        BuiltinExample{id:"unbalanced",title:"Group vector observations",program:"y = [1;2;3;4;5]; g = {'A';'A';'B';'B';'B'}; [p,tbl,s] = anova1(y,g,'off');",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(s.df==3); assert(isequal(s.n,[2 3])); assert(isequal(s.means,[1.5 4])); assert(tbl{4,3}==4); assert(p>=0 && p<=1);"}},
    ],example_exemption:None,faqs:&[],links:&[
        BuiltinDocumentationLink{label:"MATLAB anova1",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/anova1.html")},
        BuiltinDocumentationLink{label:"Implementation",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/anova1.rs")},
    ],media:&[],evidence:BuiltinDocumentationEvidence{implementation:&[],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Analytic F probabilities, grouping and table geometry",location:"builtins::stats::summary::anova1::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
