use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "group", title: "Check a group and one preference", program: "group = 'runmatCatalogIsprefExample';\nsetpref(group, 'mode', 'safe');\nassert(ispref(group));\nassert(ispref(group, 'mode'));\nassert(~ispref(group, 'missing'));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ispref(group)); assert(ispref(group, 'mode')); assert(~ispref(group, 'missing'));" } },
    BuiltinExample { id: "multiple", title: "Check several names without losing shape", program: "group = 'runmatCatalogIsprefMultiple';\nsetpref(group, 'present', 1);\nflags = ispref(group, {'present','absent'});\nassert(isequal(size(flags), [1 2]));\nassert(isequal(flags, logical([1 0])));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(flags), [1 2])); assert(isequal(flags, logical([1 0])));" } },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ispref"),
    slug: Some("ispref"),
    summary: "Test whether a preference group or named preferences exist.",
    description: "`ispref(group)` returns a scalar logical. `ispref(group, pref)` returns a scalar logical for one name or a logical array with the same shape as a string/cell name array.",
    keywords: &["ispref", "preferences", "existence"],
    related: &["getpref", "setpref"],
    sections: &[BuiltinDocumentationSection { heading: "Host controls", paragraphs: &["Group and preference names are host text. Invalid numeric or provider-resident controls reject before preference lookup or provider access.", "The in-memory preference owner does not yet supply durable cross-session storage."] }],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does checking a missing preference create it?", answer: "No. `ispref` only tests existence; use `getpref` with a default or `setpref` to create a value." }],
    links: &[BuiltinDocumentationLink { label: "getpref", target: BuiltinDocumentationLinkTarget::Builtin("getpref") }, BuiltinDocumentationLink { label: "setpref", target: BuiltinDocumentationLinkTarget::Builtin("setpref") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Preference runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar and multi-name existence with exact logical shape", location: "builtins::io::repl_fs::compat::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
