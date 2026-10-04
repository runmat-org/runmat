use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "default", title: "Create a missing preference from a default", program: "group = 'runmatCatalogGetprefExample';\nvalue = getpref(group, 'theme', 'dark');\nassert(strcmp(value, 'dark'));\nassert(strcmp(getpref(group, 'theme'), 'dark'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::ApplicationSettings(BuiltinApplicationSettingsFixture { id: BuiltinExampleFixtureId { local_name: "settings" } }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(value, 'dark')); assert(strcmp(getpref(group, 'theme'), 'dark'));" } },
    BuiltinExample { id: "multiple", title: "Read a shape-preserving name array", program: "group = 'runmatCatalogGetprefMultiple';\nsetpref(group, {'left','right'}, {uint64(1), uint64(2)});\nvalues = getpref(group, {'left','right'});\nassert(isequal(size(values), [1 2]));\nassert(isequal(values{1}, uint64(1)));\nassert(isequal(values{2}, uint64(2)));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::ApplicationSettings(BuiltinApplicationSettingsFixture { id: BuiltinExampleFixtureId { local_name: "settings" } }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(values), [1 2])); assert(isequal(values{1}, uint64(1))); assert(isequal(values{2}, uint64(2)));" } },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getpref"),
    slug: Some("getpref"),
    summary: "Read a named preference or supply a default for a missing value.",
    description: "`getpref` returns stored values without numeric-class conversion. A missing named preference without a default is an error; a supplied default creates it. Multiple names produce a matching cell array.",
    keywords: &["getpref", "preferences", "settings"],
    related: &["setpref", "ispref"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms", paragraphs: &["`getpref()` returns groups as a structure. `getpref(group, pref)` returns one value or a cell array for a name array. `getpref(group, pref, default)` creates missing values; a name array requires an equally shaped cell array of defaults.", "`getpref(group)` is a separately gated RunMat extension returning the group's values as a structure."] },
        BuiltinDocumentationSection { heading: "Storage and residency", paragraphs: &["Preferences use the application's installed settings service. Native and browser hosts can persist values across sessions; an embedding without a service reports an unavailable capability.", "Group and name controls must be host text. A provider-resident default is rejected before provider access; existing exact host integer classes and shapes are returned unchanged."] },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What if the name does not exist?", answer: "Without a default, `getpref` raises an error. With a default, the missing value is stored and returned." },
        BuiltinDocumentationFaq { question: "Are values saved across launches?", answer: "An installed durable settings host retains eligible values across launches. Explicit isolated-memory embeddings do not." },
    ],
    links: &[BuiltinDocumentationLink { label: "setpref", target: BuiltinDocumentationLinkTarget::Builtin("setpref") }, BuiltinDocumentationLink { label: "ispref", target: BuiltinDocumentationLinkTarget::Builtin("ispref") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "getpref runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences/getpref.rs") }, BuiltinDocumentationLink { label: "Shared preference controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/preferences/controls.rs") }, BuiltinDocumentationLink { label: "Application preference service", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/context/preferences.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact classes, name-array shape, defaults, and invalid controls", location: "builtins::io::repl_fs::compat::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
