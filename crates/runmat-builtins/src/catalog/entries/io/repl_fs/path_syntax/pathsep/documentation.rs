use crate::*;
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "query", title: "Read the search-path separator", program: "separator = pathsep();", display_output: Some("';' on Windows and ':' on other supported platforms"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(separator)); assert(isequal(size(separator), [1 1])); assert(strcmp(separator, ':') || strcmp(separator, ';'));" } },
    BuiltinExample { id: "path-list", title: "Recognize the separator in a path list", program: "list = ['first' pathsep() 'second'];", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(list, pathsep()));" } },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pathsep"),
    slug: Some("pathsep"),
    summary: "Return the search-path-list separator for the current platform.",
    description: "`pathsep` returns semicolon on Windows and colon on other supported platforms.",
    keywords: &["pathsep", "search path", "path separator"],
    related: &["path", "genpath", "filesep", "fullfile"],
    sections: &[BuiltinDocumentationSection {
        heading: "Search-path lists",
        paragraphs: &["`pathsep` separates complete directory names in the value returned by `path` and `genpath`. It does not separate components inside one file name; use `filesep` or `fullfile` for that purpose.", "The call accepts no inputs and returns a 1-by-1 host character row in native and browser execution."],
    }],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How is pathsep different from filesep?", answer: "`pathsep` separates directory entries in a list. `filesep` separates components within one path." }],
    links: &[BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") }, BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") }, BuiltinDocumentationLink { label: "filesep", target: BuiltinDocumentationLinkTarget::Builtin("filesep") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Separator runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/pathsep/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Platform value and arity", location: "builtins::io::repl_fs::path_syntax::pathsep::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
