use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timetable2table"),
    slug: Some("timetable2table"),
    summary: "Convert a timetable into a table.",
    description: "`timetable2table(TT)` converts a timetable to a table and places the row-time vector in the first table variable.",
    keywords: &["timetable2table", "timetable", "table", "ConvertRowTimes"],
    related: &["table2timetable", "timetable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Row times are converted by default and prepended as a `Time` variable. Set `\"ConvertRowTimes\"` to `false` to discard them.", "Variable order, native integer classes, exact payloads, and row names are preserved."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Tabular containers are host values.", "This host container operation is a fusion boundary; interactive resident inputs are gathered only after compatibility admission."] },
    ],
    examples: &[BuiltinExample { id: "row-times-to-variable", title: "Convert row times into the first table variable", program: "TT = timetable(seconds([1;2]), uint64([10;20]), 'VariableNames', {'Count'});\nT = timetable2table(TT);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T));\nassert(width(T) == 2);\nassert(isequal(T.Count, uint64([10;20])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table2timetable", target: BuiltinDocumentationLinkTarget::Builtin("table2timetable") }, BuiltinDocumentationLink { label: "timetable", target: BuiltinDocumentationLinkTarget::Builtin("timetable") }],
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
