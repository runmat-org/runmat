use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("array2timetable"),
    slug: Some("array2timetable"),
    summary: "Convert an array into a timetable.",
    description: "`array2timetable(X, timingName, timingValue)` splits a homogeneous two-dimensional array into timetable variables by column. Native integer columns preserve the exact class and values of `X`.",
    keywords: &["array2timetable", "timetable", "RowTimes", "SampleRate", "TimeStep", "StartTime", "VariableNames", "DimensionNames"],
    related: &["timetable", "array2table"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Exactly one of `\"RowTimes\"`, `\"SampleRate\"`, or `\"TimeStep\"` is required. Explicit row times must be an M-by-1 datetime or duration vector.", "`\"StartTime\"` applies to `\"SampleRate\"` and `\"TimeStep\"`; a calendar-duration step requires a datetime start. `\"VariableNames\"` and `\"DimensionNames\"` follow current table naming rules, while `\"RowNames\"` is not an `array2timetable` option.", "All eight integer classes are preserved exactly when supplied as `X` and are accepted as positive scalar sample rates.", "Interactive resident GPU arguments are mode-gated RunMat extensions. They gather exact typed storage before timetable construction and reject in MATLAB-compatible mode."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Inputs are gathered before timetable materialization.", "This host container operation is a fusion boundary; interactive resident inputs are gathered only after compatibility admission."] },
    ],
    examples: &[BuiltinExample { id: "sample-rate", title: "Split an integer array into timed columns", program: "TT = array2timetable(int16([10;20]), 'SampleRate', 2)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istimetable(TT));\nassert(height(TT) == 2);\nassert(isa(TT.Var1, 'int16'));\nassert(isequal(TT.Var1, int16([10; 20])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "timetable", target: BuiltinDocumentationLinkTarget::Builtin("timetable") }, BuiltinDocumentationLink { label: "array2table", target: BuiltinDocumentationLinkTarget::Builtin("array2table") }],
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
