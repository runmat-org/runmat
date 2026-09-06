use crate::*;
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "query", title: "Read the file separator", program: "separator = filesep();", display_output: Some("'\\' on Windows and '/' on other supported platforms"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(separator)); assert(isequal(size(separator), [1 1])); assert(strcmp(separator, '/') || strcmp(separator, char(92)));" } },
    BuiltinExample { id: "trailing", title: "Request a trailing separator", program: "folder = fullfile('data', 'raw', filesep());", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(endsWith(folder, filesep()));" } },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("filesep"),
    slug: Some("filesep"),
    summary: "Return the file-name separator for the current platform.",
    description: "`filesep` returns one character: backslash on Windows and forward slash on other supported platforms.",
    keywords: &["filesep", "file separator", "path delimiter"],
    related: &["fullfile", "fileparts", "pathsep"],
    sections: &[BuiltinDocumentationSection {
        heading: "File-name separator",
        paragraphs: &["Use `filesep` when code needs the platform delimiter as data. Prefer `fullfile` when assembling paths because it also handles repeated delimiters and text containers.", "`filesep` accepts no inputs, returns a 1-by-1 character row, and is available in native and browser execution."],
    }],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How is filesep different from pathsep?", answer: "`filesep` separates components within one file name. `pathsep` separates entries in a search-path list." }],
    links: &[BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") }, BuiltinDocumentationLink { label: "pathsep", target: BuiltinDocumentationLinkTarget::Builtin("pathsep") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Separator runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/filesep/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Platform value and arity", location: "builtins::io::repl_fs::path_syntax::filesep::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
