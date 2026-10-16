use super::super::shared::{SOURCE, TABLE_TEST};
use crate::*;

pub(super) const SETVAROPTS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setvaropts"), slug: Some("setvaropts"),
    summary: "Set supported controls for selected import variables.",
    description: "`setvaropts` returns a new typed import-options object whose selected variable controls are consumed by `readtable` and, for numeric text, `readmatrix`. The input object remains unchanged.",
    keywords: &["setvaropts", "detectImportOptions", "readtable", "readmatrix", "TreatAsMissing", "ThousandsSeparator", "InputFormat"],
    related: &["setvartype", "getvaropts", "detectImportOptions", "readtable", "readmatrix"],
    sections: &[BuiltinDocumentationSection { heading: "Supported controls", paragraphs: &[
        "Selection accepts exact variable names, one-based numeric indices, integer arrays, or logical masks. Omit selection to apply the pairs to every variable. Supported controls are TreatAsMissing, DecimalSeparator, ThousandsSeparator, FillValue, EmptyFieldRule (`missing` or `error`), WhitespaceRule (`trim`), and datetime InputFormat in the supported numeric date-token grammar. Unknown controls and unsupported rule values fail before reader I/O.",
        "Per-variable missing, fill, and numeric separators affect actual table columns and numeric matrix fields. InputFormat uses the same datetime parser as the datetime builtin for a variable explicitly typed datetime. Exact wide integer table values remain on the existing native integer text conversion path.",
        "This is partial MATLAB compatibility: type-specific VariableOptions subclasses, the full set of whitespace/empty-field rules, and several specialized import controls are not yet implemented. Spreadsheet import is limited to the existing spreadsheet reader capabilities; unsupported combinations are rejected rather than accepted as inert metadata.",
    ] }],
    examples: &[BuiltinExample {
        id: "missing-column", title: "Replace a per-column missing token",
        program: "opts = detectImportOptions('force.csv');\nopts = setvartype(opts,'Force_N','double');\nopts = setvaropts(opts,'Force_N','TreatAsMissing','N/A','FillValue',-1);\nT = readtable('force.csv',opts);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "missing-column" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "force.csv", content: BuiltinFixtureContent::Utf8("Force_N\nN/A\n2\n") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(T.Force_N,[-1;2]));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "readtable", target: BuiltinDocumentationLinkTarget::Builtin("readtable") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TABLE_TEST], notes: &["Source packet only until compiled reader and example execution qualify the form."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
