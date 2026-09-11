use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "System temporary directory", paragraphs: &["`tempdir()` returns the temporary directory selected for the current RunMat session as a nonempty character row. The result ends with the platform file separator, so existing code may concatenate a filename directly.", "The function reports a location; it does not create, reserve, or clean files. Native sessions use the host temporary-directory policy. Browser sessions use the corresponding sandbox filesystem location."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The result is host text and the operation never invokes an acceleration provider. `tempdir` accepts no inputs and is available in interpreted, JIT-compiled, AOT-compiled, native, and browser execution."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "query", title: "Read the temporary directory", program: "folder = tempdir();", display_output: Some("folder is a nonempty character row ending in the platform separator"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(folder)); assert(size(folder, 1) == 1); assert(~isempty(folder)); assert(endsWith(folder, '/') || endsWith(folder, char(92)));" } },
    BuiltinExample { id: "stable", title: "Use the session location consistently", program: "first = tempdir();\nsecond = tempdir();", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(first, second));" } },
    BuiltinExample { id: "join", title: "Build a path below the temporary directory", program: "target = fullfile(tempdir(), 'runmat-session.log');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(target)); assert(contains(target, 'runmat-session.log'));" } },
    BuiltinExample { id: "temporary-folder", title: "Create a temporary working folder", program: "work = tempname();\nmkdir(work);\ncreated = exist(work, 'dir');\nrmdir(work);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(created == 7); assert(exist(work, 'dir') == 0);" } },
    BuiltinExample { id: "nested-path", title: "Describe a task-specific path", program: "work = fullfile(tempdir(), 'runmat-task', 'results.mat');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(work)); assert(contains(work, 'runmat-task')); assert(endsWith(work, 'results.mat'));" } },
    BuiltinExample { id: "display", title: "Display the selected directory", program: "folder = tempdir();\nfprintf('RunMat temp folder: %s\\n', folder);", display_output: Some("Prints the selected temporary directory"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(folder)); assert(~isempty(folder));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why is the result a character vector?", answer: "The character-row result preserves compatibility with code that concatenates a filename onto `tempdir`. Use `string(tempdir())` when a string scalar is more convenient." },
    BuiltinDocumentationFaq { question: "Does tempdir create or clean files?", answer: "No. It only returns the session's temporary-directory path." },
    BuiltinDocumentationFaq { question: "Does the result include a trailing separator?", answer: "Yes. RunMat appends the platform separator when the selected path does not already end with one." },
    BuiltinDocumentationFaq { question: "Can a GPU accelerate tempdir?", answer: "No. Resolving a session path is a host operation and has no GPU inputs or kernels." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempdir/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tempdir"),
    slug: Some("tempdir"),
    summary: "Return the current session's temporary-directory path.",
    description: "`tempdir` returns a character row containing the temporary directory with a trailing platform separator.",
    keywords: &["tempdir", "temporary directory", "temporary path", "filesep"],
    related: &["tempname", "fullfile", "mkdir", "rmdir"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Temporary-directory runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempdir/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Path form, separators, arity, and session stability", location: "builtins::io::repl_fs::temporary_path::tempdir::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
