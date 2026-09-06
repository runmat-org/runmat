use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Lexical path parsing", paragraphs: &["`fileparts(filename)` separates path text into its folder, base name, and extension. It does not access the filesystem and the input does not need to exist.", "A character row produces character-row outputs. A string array produces string arrays of the same shape. A cell array of character rows produces cell arrays of the same shape."] },
    BuiltinDocumentationSection { heading: "Names and extensions", paragraphs: &["The extension begins at the final dot after the last path delimiter. A leading-dot filename such as `.profile` has an empty base name and `.profile` as its extension. A trailing path delimiter denotes a folder and therefore produces empty name and extension outputs.", "On Windows, both slash directions are recognized as delimiters. On other platforms, `/` is the path delimiter and a backslash may be part of a filename."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "basic", title: "Split a file path", program: "[folder, name, ext] = fileparts(fullfile('data', 'sample.csv'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(folder, 'data')); assert(strcmp(name, 'sample')); assert(strcmp(ext, '.csv'));" } },
    BuiltinExample { id: "multiple-dots", title: "Split the final extension", program: "[folder, name, ext] = fileparts('archive.part.tar');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(folder)); assert(strcmp(name, 'archive.part')); assert(strcmp(ext, '.tar'));" } },
    BuiltinExample { id: "dotfile", title: "Parse a leading-dot filename", program: "[folder, name, ext] = fileparts(fullfile('home', '.profile'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(folder, 'home')); assert(isempty(name)); assert(strcmp(ext, '.profile'));" } },
    BuiltinExample { id: "string-array", title: "Preserve a string-array shape", program: "files = [\"src/first.m\" \"test/second.m\"];\n[folders, names, extensions] = fileparts(files);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(names)); assert(isequal(size(names), size(files))); assert(names(1) == \"first\"); assert(names(2) == \"second\"); assert(all(extensions == \".m\"));" } },
    BuiltinExample { id: "cell-array", title: "Preserve a cell-array shape", program: "files = {'src/first.m'; 'test/second.txt'};\n[folders, names, extensions] = fileparts(files);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(names)); assert(isequal(size(names), size(files))); assert(strcmp(names{1}, 'first')); assert(strcmp(extensions{2}, '.txt'));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does fileparts check whether the path exists?",
        answer: "No. It parses path text only.",
    },
    BuiltinDocumentationFaq {
        question: "Why does the extension include a dot?",
        answer:
            "Keeping the leading dot allows `strcat(name, ext)` to reconstruct the final filename.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "fullfile",
        target: BuiltinDocumentationLinkTarget::Builtin("fullfile"),
    },
    BuiltinDocumentationLink {
        label: "filesep",
        target: BuiltinDocumentationLinkTarget::Builtin("filesep"),
    },
    BuiltinDocumentationLink {
        label: "pathsep",
        target: BuiltinDocumentationLinkTarget::Builtin("pathsep"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fileparts"),
    slug: Some("fileparts"),
    summary: "Split path text into folder, base filename, and extension components.",
    description: "`fileparts` lexically parses character, string, and cell path containers while preserving representation and shape.",
    keywords: &["fileparts", "path", "filename", "extension"],
    related: &["fullfile", "filesep", "pathsep"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Lexical path runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/fileparts/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, cell, dotfile, and trailing-delimiter behavior", location: "builtins::io::repl_fs::path_syntax::fileparts::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
