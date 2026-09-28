use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timetable"),
    slug: Some("timetable"),
    summary: "Create a timetable from row times and variables.",
    description: "`timetable` groups same-height data variables with a datetime or duration row-time vector. Row times can be supplied directly or generated from `SampleRate` or `TimeStep`.",
    keywords: &["timetable", "RowTimes", "table", "time series"],
    related: &["table2timetable", "istimetable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Variables must have the same number of rows as the row-time vector. Integer variables retain their native class and payload.", "Use `RowTimes`, `SampleRate`, or `TimeStep` to choose one timing source. `StartTime` can accompany `SampleRate` or `TimeStep`.", "Preallocation accepts `Size`, `VariableTypes`, and a timing source. Integer variable types allocate native integer columns directly.", "Numeric row times and generated sequential row times without a timing source are available only in `runmat` compatibility mode.", "Table indexing methods are shared with table objects."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Timetables are host containers.", "This host container operation is a fusion boundary; interactive resident inputs are gathered only after compatibility admission."] },
    ],
    examples: &[BuiltinExample { id: "row-times-and-integer-variable", title: "Create a timetable with explicit row times", program: "times = datetime(2026, 7, [1; 2]);\ncounts = uint64([10; 20]);\nTT = timetable(times, counts, 'VariableNames', {'Count'})", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istimetable(TT));\nassert(height(TT) == 2);\nassert(isa(TT.Count, 'uint64'));\nassert(isequal(TT.Count, uint64([10; 20])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table2timetable", target: BuiltinDocumentationLinkTarget::Builtin("table2timetable") }, BuiltinDocumentationLink { label: "istimetable", target: BuiltinDocumentationLinkTarget::Builtin("istimetable") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/timetable.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Timetable semantics and integer preservation", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM tabular integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }],
        notes: &[
            "Legacy title, category, keywords, summary, description, behaviors, links, and source are preserved in catalog documentation. The original example now has deterministic assertions.",
            "Legacy gpu_support and fusion fields are represented by typed placement and the execution section. requires_feature was null; no optional feature is introduced. The tested field maps to verification evidence.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
