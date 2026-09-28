use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("readtimetable"),
    slug: Some("readtimetable"),
    summary: "Read tabular data into a timetable.",
    description: "`readtimetable(filename, ...)` imports column-oriented file data and designates the first datetime or duration variable as the timetable row times.",
    keywords: &["readtimetable", "timetable", "readtable", "RowTimes"],
    related: &["readtable", "table2timetable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Text and spreadsheet options accepted by `readtable` are forwarded, including exact native integer variable types and the documented all-eight-class `VariableNamesLine` control.", "Typed-integer values for otherwise floating-only import controls are RunMat-only extensions and are rejected in MATLAB compatibility mode before gathering or file access.", "The public contract infers row times from the first imported datetime or duration variable. RunMat extension mode additionally accepts a numeric `RowTimes` column and preserves native integer values exactly.", "If no datetime or duration variable exists, sequential numeric row times are generated only in RunMat extension mode; MATLAB-compatible mode reports the missing time source."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["File import runs on the host.", "File import uses the active filesystem and is a host-side fusion boundary. Typed extension controls are admitted before any gather or file access."] },
    ],
    examples: &[BuiltinExample { id: "csv-row-times", title: "Import datetime row times from CSV", program: "TT = readtimetable('measurements.csv', 'VariableTypes', [\"datetime\" \"uint64\"]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "timetable-csv" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "measurements.csv", content: BuiltinFixtureContent::Utf8("Time,Count\n2026-07-01,10\n2026-07-02,20\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istimetable(TT));\nassert(height(TT) == 2);\nassert(isa(TT.Count, 'uint64'));\nassert(isequal(TT.Count, uint64([10;20])));" } },
        BuiltinExample { id: "exact-integer-variable", title: "Retain an imported uint64 column", program: "TT = readtimetable('measurements.csv', 'VariableTypes', [\"datetime\" \"uint64\"]);\ncounts = TT.Count;", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "timetable-csv" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "measurements.csv", content: BuiltinFixtureContent::Utf8("Time,Count\n2026-07-01,10\n2026-07-02,20\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(counts, 'uint64'));\nassert(isequal(counts, uint64([10;20])));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "readtable", target: BuiltinDocumentationLinkTarget::Builtin("readtable") }, BuiltinDocumentationLink { label: "table2timetable", target: BuiltinDocumentationLinkTarget::Builtin("table2timetable") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/timetable.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Timetable semantics and integer preservation", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }],
        notes: &[
            "Legacy title, category, keywords, summary, description, behaviors, links, and source are preserved in catalog documentation. The 2 example scenarios now have deterministic assertions and isolated CSV fixture.",
            "Legacy gpu_support and fusion fields are represented by typed placement and the execution section. requires_feature was null; no optional feature is introduced. The tested field maps to verification evidence.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
