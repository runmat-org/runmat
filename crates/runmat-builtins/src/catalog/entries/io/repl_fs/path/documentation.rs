use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Read or replace the session search path",
        paragraphs: &[
            "`path` with no input returns the ordered search-path entries as a character row vector. `path(path1)` replaces the list, while `path(path1, path2)` joins the two fragments with the platform path separator before replacing it. Mutation forms return the previous path.",
            "The current working folder is searched before these explicit entries and is not included in the returned text. An empty replacement clears only the explicit entries.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resolution and supported input forms",
        paragraphs: &[
            "Changes apply immediately to later function, script, class, package, MEX, and file discovery in the same RunMat session. Each session owns its path state, so one session does not change another session's resolution order.",
            "Portable code should pass character rows or string scalars. RunMat compatibility mode also accepts a dense real numeric row of Unicode scalar values. A resident numeric row is admitted by compatibility policy before it is gathered; the resulting path and return value remain host text.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "query", title: "Read the current search path", program: "p = path;\nassert(ischar(p));\nassert(size(p, 1) == 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(p)); assert(size(p, 1) == 1);" } },
    BuiltinExample { id: "temporary", title: "Temporarily replace and restore the search path", program: "original = path;\nold = path('runmat-addons');\nassert(strcmp(old, original));\nassert(strcmp(path, 'runmat-addons'));\npath(old);\nassert(strcmp(path, original));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "append", title: "Append an entry", program: "original = path;\nextra = 'projects/analysis';\npath(original, extra);\nassert(contains(path, extra));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "prepend", title: "Prepend an entry", program: "original = path;\nextra = 'toolbox';\npath(extra, original);\nassert(startsWith(path, extra));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "generated", title: "Prepend a recursively generated folder list", program: "mkdir('submodules');\nmkdir(fullfile('submodules', 'tooling'));\noriginal = path;\ntooling = genpath(fullfile('submodules', 'tooling'));\nold = path(tooling, original);\nassert(strcmp(old, original));\nassert(contains(path, 'tooling'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "numeric-codes", title: "Set a path from numeric character codes in RunMat mode", program: "original = path;\ncodes = uint16('extension-path');\npath(codes);\nassert(strcmp(path, 'extension-path'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does path include the current folder?", answer: "No. The current folder is an implicit first search location; `path` returns only the explicit ordered entries." },
    BuiltinDocumentationFaq { question: "Can I clear the explicit path?", answer: "Yes. `path('')` clears the explicit entries while leaving the current folder available for resolution." },
    BuiltinDocumentationFaq { question: "How do I append or prepend?", answer: "Use `path(path, newEntry)` to append or `path(newEntry, path)` to prepend. `addpath` provides validated folder-oriented forms for the same common operations." },
    BuiltinDocumentationFaq { question: "Do later calls see a change immediately?", answer: "Yes. Runtime callable and file discovery read the active session's ordered path for each resolution request." },
    BuiltinDocumentationFaq { question: "Does one session change another session's path?", answer: "No. Active RunMat sessions own independent path state. Calls made without a session use the process compatibility state." },
    BuiltinDocumentationFaq { question: "Can path accept numeric character codes?", answer: "RunMat mode accepts a dense real numeric row as a convenience extension. MATLAB compatibility modes require character or string text." },
];

const RELATED: &[&str] = &[
    "addpath", "rmpath", "genpath", "which", "exist", "cd", "copyfile", "delete", "dir",
    "fullfile", "getenv", "ls", "mkdir", "movefile", "pwd", "rmdir", "savepath", "setenv",
    "tempdir", "tempname",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Builtin("which") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "savepath", target: BuiltinDocumentationLinkTarget::Builtin("savepath") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("path"), slug: Some("path"),
    summary: "Query or replace the active session search path.",
    description: "`path` reads or replaces the ordered path used for runtime function, script, class, package, MEX, and file resolution.",
    keywords: &["path", "search path", "function resolution", "session path", "pathsep"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Search-path runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Query, mutation, admission, isolation, and generation", location: "builtins::io::repl_fs::path::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Runtime callable resolution after path mutation", location: "runmat-core path-state tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native filesystem and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
