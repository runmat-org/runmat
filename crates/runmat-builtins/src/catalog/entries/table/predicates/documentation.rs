use crate::*;

const PLACEMENT: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Execution and placement",
    paragraphs: &["The predicate reads runtime class metadata and returns one host logical scalar. It does not gather array payloads, including GPU-resident integer arrays, and is a fusion boundary."],
};
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/predicates.rs"),
};
const TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Tabular and categorical predicate tests",
    location: "crates/runmat-runtime/src/builtins/table/tests.rs",
};
const INTEGRATION: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::IntegrationTest,
    label: "VM table and timetable predicate integration",
    location: "crates/runmat-vm/tests/tables_timetables.rs",
};

pub(super) const ISCATEGORICAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("iscategorical"),
    slug: Some("iscategorical"),
    summary: "Return true for categorical arrays.",
    description: "`iscategorical(A)` returns true when `A` has the categorical runtime class.",
    keywords: &["iscategorical", "categorical", "predicate"],
    related: &["categorical", "ordinal", "isordinal"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Non-categorical values return false. Ordinal and nominal categorical values both return true."] }, PLACEMENT],
    examples: &[BuiltinExample { id: "categorical-and-numeric", title: "Distinguish categorical values", program: "C = categorical({'a'}); tf = iscategorical(C);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true)); assert(isequal(iscategorical(1), false));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "categorical", target: BuiltinDocumentationLinkTarget::Builtin("categorical") }, BuiltinDocumentationLink { label: "ordinal", target: BuiltinDocumentationLinkTarget::Builtin("ordinal") }, BuiltinDocumentationLink { label: "isordinal", target: BuiltinDocumentationLinkTarget::Builtin("isordinal") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TEST, INTEGRATION], notes: &["Legacy behavior, links, source and placement notes are retained; the display-only example now asserts positive and negative results.", "Legacy requires_feature was null; tested maps to unit and integration evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(super) const ISORDINAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isordinal"),
    slug: Some("isordinal"),
    summary: "Return true for ordinal categorical arrays.",
    description: "`isordinal(A)` returns true when `A` is categorical and its `Ordinal` metadata flag is true.",
    keywords: &["isordinal", "ordinal", "categorical", "predicate"],
    related: &["ordinal", "categorical", "iscategorical"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Ordinal categorical values created with `ordinal` or `categorical(..., 'Ordinal', true)` return true.", "Nominal categorical values and all non-categorical values return false."] }, PLACEMENT],
    examples: &[BuiltinExample { id: "ordinal-metadata", title: "Inspect ordinal categorical metadata", program: "C = ordinal({'low'}); tf = isordinal(C);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true)); assert(isequal(isordinal(categorical({'low'})), false)); assert(isequal(isordinal(1), false));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "ordinal", target: BuiltinDocumentationLinkTarget::Builtin("ordinal") }, BuiltinDocumentationLink { label: "categorical", target: BuiltinDocumentationLinkTarget::Builtin("categorical") }, BuiltinDocumentationLink { label: "iscategorical", target: BuiltinDocumentationLinkTarget::Builtin("iscategorical") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TEST, INTEGRATION], notes: &["Legacy behavior, links, source and placement notes are retained; the display-only example now asserts positive and negative results.", "Legacy requires_feature was null; tested maps to unit and integration evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(super) const ISTABLE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("istable"),
    slug: Some("istable"),
    summary: "Return true for table objects.",
    description: "`istable(A)` returns true when `A` has the table runtime class.",
    keywords: &["istable", "table", "predicate"],
    related: &["table", "istimetable"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Timetables have a distinct runtime class and return false from `istable`.", "Non-table values return false."] }, PLACEMENT],
    examples: &[BuiltinExample { id: "table-and-numeric", title: "Identify a table", program: "T = table([1;2]); tf = istable(T);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true)); assert(isequal(istable([1;2]), false));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "istimetable", target: BuiltinDocumentationLinkTarget::Builtin("istimetable") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TEST, INTEGRATION], notes: &["Legacy behavior and link are retained. The obsolete table/mod.rs source link is replaced by the owning predicate implementation.", "Legacy requires_feature was null; tested maps to unit and integration evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(super) const ISTIMETABLE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("istimetable"),
    slug: Some("istimetable"),
    summary: "Return true for timetable objects.",
    description: "`istimetable(A)` returns true when `A` has the timetable runtime class.",
    keywords: &["istimetable", "timetable", "predicate"],
    related: &["timetable", "istable"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Plain tables and non-tabular values return false."] }, PLACEMENT],
    examples: &[BuiltinExample { id: "timetable-and-table", title: "Identify a timetable", program: "TT = timetable(datetime(2026,7,[1;2]), [10;20]); tf = istimetable(TT);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true)); assert(isequal(istimetable(table([10;20])), false));" } }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "timetable", target: BuiltinDocumentationLinkTarget::Builtin("timetable") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TEST, INTEGRATION], notes: &["Legacy behavior and link are retained. The obsolete table/mod.rs source link is replaced by the owning predicate implementation.", "Legacy requires_feature was null; tested maps to unit and integration evidence."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
