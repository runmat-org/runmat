use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Generate a temporary path", paragraphs: &["`tempname()` selects a path below `tempdir`. `tempname(folder)` selects one below the supplied character row or string scalar. Absolute and relative folders are retained, and a leading `~` is expanded through the session environment.", "The returned character row names a path that did not exist when RunMat checked it. The call does not create or reserve the file or directory, so code that shares the filesystem must still create the resource safely before another process can claim it."] },
    BuiltinDocumentationSection { heading: "Uniqueness and execution", paragraphs: &["Candidates combine session time, a process identifier where available, and a process-wide counter. RunMat checks the portable filesystem before returning a candidate and retries a bounded number of times after collisions. The token is intended for temporary-path uniqueness, not as a cryptographic secret.", "Path selection runs on the host or browser sandbox and never invokes an acceleration provider. Nontext and provider-resident inputs reject before filesystem access."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "default", title: "Generate a temporary path", program: "name = tempname();", display_output: Some("name is a nonexisting character-row path below tempdir"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(name)); assert(size(name, 1) == 1); assert(startsWith(name, tempdir())); assert(exist(name, 'file') == 0); assert(exist(name, 'dir') == 0);" } },
    BuiltinExample { id: "unique", title: "Request two distinct names", program: "first = tempname();\nsecond = tempname();", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~strcmp(first, second));" } },
    BuiltinExample { id: "folder", title: "Select a path in a chosen folder", program: "mkdir('scratch');\nname = tempname('scratch');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(startsWith(name, 'scratch')); assert(exist(name, 'file') == 0);" } },
    BuiltinExample { id: "extension", title: "Append a filename extension", program: "csv_path = [tempname() '.csv'];", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(endsWith(csv_path, '.csv')); assert(exist(csv_path, 'file') == 0);" } },
    BuiltinExample { id: "create-directory", title: "Create the selected directory", program: "scratch = tempname();\nmkdir(scratch);\ncreated = exist(scratch, 'dir');\nrmdir(scratch);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(created == 7); assert(exist(scratch, 'dir') == 0);" } },
    BuiltinExample { id: "future-folder", title: "Select a name below a folder that does not exist yet", program: "folder = 'future-temp-folder';\nname = tempname(folder);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(startsWith(name, folder)); assert(exist(name, 'file') == 0); assert(exist(name, 'dir') == 0);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does tempname create the file or folder?", answer: "No. It returns a candidate that was unused when checked. Create the resource with the operation appropriate to your program." },
    BuiltinDocumentationFaq { question: "Is the name guaranteed to remain available?", answer: "No path-only API can reserve a name atomically. Another process can create it after the check. Use an atomic create operation when that race matters." },
    BuiltinDocumentationFaq { question: "Can the folder be relative or missing?", answer: "Yes. RunMat joins the token to the supplied folder without creating or requiring that folder." },
    BuiltinDocumentationFaq { question: "Is the token suitable for secrets?", answer: "No. It is designed to avoid ordinary temporary-path collisions, not to provide cryptographic randomness." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "fopen", target: BuiltinDocumentationLinkTarget::Builtin("fopen") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempname/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tempname"),
    slug: Some("tempname"),
    summary: "Generate an unused temporary path in the default or a selected folder.",
    description: "`tempname` returns a character-row path without creating the corresponding file or directory.",
    keywords: &["tempname", "temporary file", "unique path", "temporary directory"],
    related: &["tempdir", "fopen", "mkdir", "fullfile", "delete"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Temporary-name runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/temporary_path/tempname/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Admission, folder selection, uniqueness, and browser behavior", location: "builtins::io::repl_fs::temporary_path::tempname::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
