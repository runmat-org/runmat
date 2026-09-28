use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pivot"),
    slug: Some("pivot"),
    summary: "Pivot table data into a wide grouped summary table.",
    description: "`pivot(T, rowvars, colvars, datavar)` groups input rows, expands observed column groups into variables, and summarizes one numeric data column for each row/column combination.",
    keywords: &["pivot", "table", "groupsummary", "reshape"],
    related: &["groupsummary"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Row, column, and data variables use table variable selector semantics. `datavar` must select exactly one numeric column vector. Group values are observed in input order; missing row/column combinations yield NaN.",
            "`'Method', method` selects the summary method and defaults to `'sum'`. Output variable names combine the column-key label with the data variable name and are made valid and unique.",
            "Integer grouping keys retain their exact values, including wide integers. Summary columns are presently constructed as double values; a method's numerical conversion boundary remains explicit in the runtime.",
            "The newer named `Rows`, `Columns`, `DataVariable`, binning, and totals forms are not implemented. Unsupported name-value options reject rather than silently changing the grouping.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Pivoting runs on the host. Inputs that require residency transfer are gathered before table grouping; pivot does not run a GPU kernel or fuse with a numeric operation.",
            "The result is a host table. Grouping cardinality and table contents are runtime data, so static analysis preserves the output table class but does not invent a column schema.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "grouped-sum", title: "Expand groups into summary columns",
        program: "Group = [\"A\";\"A\";\"B\"]; Kind = [\"x\";\"y\";\"x\"]; Value = [1;2;3]; T = table(Group,Kind,Value,'VariableNames',{'Group','Kind','Value'}); P = pivot(T, 'Group', 'Kind', 'Value');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(istable(P)); assert(height(P)==2); assert(width(P)==3); assert(isequal(P.x_Value,[1;3])); assert(P.y_Value(1)==2); assert(isnan(P.y_Value(2)));" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "groupsummary", target: BuiltinDocumentationLinkTarget::Builtin("groupsummary") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/object/analytics.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pivot wide-summary table", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }],
        notes: &["Example assertions check output contents and shape.", "Pivot grouping runs on the host and forms a fusion boundary.", "Binning and totals forms that the current implementation rejects are documented as limitations rather than supported capabilities.", "The linked tests cover the supported host forms."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
