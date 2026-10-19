use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table2timetable"),
    slug: Some("table2timetable"),
    summary: "Convert a table into a timetable.",
    description: "`table2timetable(T)` converts a table to a timetable by selecting or supplying datetime or duration row times. Regular row times can instead be generated from `SampleRate`, `TimeStep`, and an optional `StartTime`.",
    keywords: &["table2timetable", "table", "timetable", "RowTimes"],
    related: &["timetable2table"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Without timing options, the first datetime or duration table variable becomes row times and is removed from the data-variable list.",
                "`RowTimes` accepts a datetime or duration vector, a variable name, or a positive one-based numeric variable index. Typed-integer indices are decoded exactly in every supported integer class.",
                "RunMat extension mode also accepts a numeric row-time column and preserves native integer storage exactly; MATLAB-compatible mode requires datetime or duration row-time values.",
                "A positive numeric `SampleRate`, including any built-in integer class, generates duration row times through the documented floating timing boundary.",
                "Integer data variables retain their exact classes and values independently of row-time construction.",
                "When no time variable or timing option exists, generated sequential row times are available only in RunMat extension mode. Converting table row names into a timetable data variable remains an implementation gap.",
            ],
        },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Tabular containers are host values.", "This host container operation is a fusion boundary; interactive resident inputs are gathered only after compatibility admission."] },
    ],
    examples: &[BuiltinExample { id: "explicit-row-times", title: "Supply a duration row-time vector", program: "T = table(uint64([10;20]), 'VariableNames', {'Count'});\ntimes = seconds([1;2]);\nTT = table2timetable(T, 'RowTimes', times);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istimetable(TT));\nassert(height(TT) == 2);\nassert(isequal(TT.Count, uint64([10;20])));" } },
        BuiltinExample { id: "time-variable-index", title: "Select a time variable by exact integer index", program: "T = table(seconds([1;2]), uint64([10;20]), 'VariableNames', {'Time','Count'});\nTT = table2timetable(T, 'RowTimes', uint8(1));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istimetable(TT));\nassert(width(TT) == 1);\nassert(isequal(TT.Count, uint64([10;20])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "timetable2table", target: BuiltinDocumentationLinkTarget::Builtin("timetable2table") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/timetable.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Timetable semantics and integer preservation", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM tabular integration", location: "crates/runmat-vm/tests/tables_timetables.rs" }],
        notes: &[
            "Legacy title, category, keywords, summary, description, behaviors, links, and source are preserved in catalog documentation. The 2 example scenarios now have deterministic assertions.",
            "Legacy gpu_support and fusion fields are represented by typed placement and the execution section. requires_feature was null; no optional feature is introduced. The tested field maps to verification evidence.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
