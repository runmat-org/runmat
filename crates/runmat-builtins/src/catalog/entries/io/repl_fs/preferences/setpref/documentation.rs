use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "integer", title: "Store an exact integer", program: "group = 'runmatCatalogSetprefInteger';\nsetpref(group, 'maximum', uint64(42));\nassert(isequal(getpref(group, 'maximum'), uint64(42)));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(getpref(group, 'maximum'), uint64(42)));" } },
    BuiltinExample { id: "multiple", title: "Assign several names together", program: "group = 'runmatCatalogSetprefMultiple';\nsetpref(group, {'first','second'}, {int64(-7), 'ready'});\nvalues = getpref(group, {'first','second'});\nassert(isequal(values{1}, int64(-7)));\nassert(strcmp(values{2}, 'ready'));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values{1}, int64(-7))); assert(strcmp(values{2}, 'ready'));" } },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setpref"),
    slug: Some("setpref"),
    summary: "Store one or more named preferences in memory.",
    description: "`setpref(group, pref, value)` preserves the class and shape of stored values. For multiple names, provide an equally shaped cell array of values. The documented statement form has no result; a numeric status is a RunMat extension.",
    keywords: &["setpref", "preferences", "settings"],
    related: &["getpref", "ispref"],
    sections: &[
        BuiltinDocumentationSection { heading: "Controls and values", paragraphs: &["Group and name controls are host text. All name/value cardinality and shape checks occur before a provider download or store mutation.", "Top-level provider-resident values gather through their owning provider before committing the batch. Exact integer classes and shapes are retained."] },
        BuiltinDocumentationSection { heading: "Storage lifetime", paragraphs: &["Preferences are currently kept in memory, not saved across RunMat launches."] },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can I set several preferences at once?", answer: "Yes. Supply a string/cell array of names and an equally shaped cell array of values." }, BuiltinDocumentationFaq { question: "Does the value change numeric precision?", answer: "No. Host integer values retain their native class and payload; a top-level resident value is gathered without floating conversion." }],
    links: &[BuiltinDocumentationLink { label: "getpref", target: BuiltinDocumentationLinkTarget::Builtin("getpref") }, BuiltinDocumentationLink { label: "ispref", target: BuiltinDocumentationLinkTarget::Builtin("ispref") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "setpref runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences/setpref.rs") }, BuiltinDocumentationLink { label: "Shared preference controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences/controls.rs") }, BuiltinDocumentationLink { label: "Current preference storage", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences/store.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact integer values, name/value shapes, and rejection before gather", location: "builtins::io::repl_fs::compat::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
