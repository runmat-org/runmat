use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Persist the current search path",
        paragraphs: &[
            "`savepath(filename)` writes a `pathdef` function containing the current ordered RunMat search path. It does not modify the active session path. The returned status is `0` on success and `1` when the target cannot be resolved or written.",
            "A character vector or string scalar may name a relative or absolute file. The generated file returns the exact path-list character row, with apostrophes escaped for MATLAB-syntax source.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Default target and RunMat extensions",
        paragraphs: &[
            "With no filename, RunMat uses `RUNMAT_PATHDEF` when it is set. Otherwise it writes `$HOME/.runmat/pathdef.m` on Linux and macOS or `%USERPROFILE%\\.runmat\\pathdef.m` on Windows. RunMat mode creates a missing parent folder for the target.",
            "RunMat mode also accepts a directory target and appends `pathdef.m`, accepts dense real numeric character-code rows, and can return `[status, message, messageID]`. MATLAB compatibility modes retain the documented zero/one-input and single-status-output surface and reject these extensions before provider or filesystem work.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "explicit", title: "Write an explicit pathdef file", program: "status = savepath('project_pathdef.m');\nassert(status == 0);\nsource = fileread('project_pathdef.m');\nassert(contains(source, 'function p = pathdef'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(contains(source, 'function p = pathdef'));" } },
    BuiltinExample { id: "command", title: "Use command syntax", program: "savepath saved_pathdef.m\nassert(exist('saved_pathdef.m', 'file') == 2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(exist('saved_pathdef.m', 'file') == 2);" } },
    BuiltinExample { id: "ordered", title: "Persist the current ordered path", program: "mkdir('first');\nmkdir('second');\noriginal = path;\naddpath('first', '-end');\naddpath('second', '-end');\nstatus = savepath('ordered_pathdef.m');\nsource = fileread('ordered_pathdef.m');\npath(original);\nassert(status == 0);\nassert(contains(source, 'first'));\nassert(contains(source, 'second'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(contains(source, 'first')); assert(contains(source, 'second'));" } },
    BuiltinExample { id: "directory", title: "Use a directory target in RunMat mode", program: "mkdir('profiles');\nstatus = savepath('profiles');\nassert(status == 0);\nassert(exist(fullfile('profiles', 'pathdef.m'), 'file') == 2);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(exist(fullfile('profiles', 'pathdef.m'), 'file') == 2);" } },
    BuiltinExample { id: "diagnostics", title: "Request RunMat diagnostic outputs", program: "[status, message, messageID] = savepath('diagnostic_pathdef.m');\nassert(status == 0);\nassert(isempty(message));\nassert(isempty(messageID));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(isempty(message)); assert(isempty(messageID));" } },
    BuiltinExample { id: "numeric-codes", title: "Use exact numeric filename codes in RunMat mode", program: "filename = uint16('encoded_pathdef.m');\nstatus = savepath(filename);\nassert(status == 0);\nassert(exist('encoded_pathdef.m', 'file') == 2);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(exist('encoded_pathdef.m', 'file') == 2);" } },
    BuiltinExample { id: "configured-default", title: "Configure the default target", program: "target = fullfile(pwd, 'configured_pathdef.m');\nsetenv('RUNMAT_PATHDEF', target);\nstatus = savepath();\nassert(status == 0);\nassert(exist(target, 'file') == 2);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(exist(target, 'file') == 2);" } },
    BuiltinExample { id: "execute-generated", title: "Execute the generated pathdef function", program: "original = path;\nstatus = savepath('pathdef.m');\nassert(status == 0);\nsaved = pathdef();\nassert(strcmp(saved, original));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(strcmp(saved, original));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does savepath change the current path?", answer: "No. It reads the active ordered path and writes a source file; use `path`, `addpath`, or `rmpath` to change the session." },
    BuiltinDocumentationFaq { question: "What does the generated file contain?", answer: "It defines `pathdef`, which returns the saved platform path-list character row." },
    BuiltinDocumentationFaq { question: "Does savepath create missing folders?", answer: "RunMat mode creates a missing parent folder. MATLAB compatibility modes report status `1` when the parent does not exist." },
    BuiltinDocumentationFaq { question: "Can I pass a directory instead of a filename?", answer: "In RunMat mode, yes; `pathdef.m` is appended. Compatibility modes require an explicit file path." },
    BuiltinDocumentationFaq { question: "Does a GPU accelerate savepath?", answer: "No. Filesystem persistence is host-owned. A RunMat-only resident numeric character-code row is gathered once after compatibility and shape admission." },
    BuiltinDocumentationFaq { question: "Can another MATLAB-syntax runtime execute the generated file?", answer: "Yes. The generated source defines a `pathdef` function that returns the saved platform path-list character row." },
    BuiltinDocumentationFaq { question: "How do I restore a saved path?", answer: "Make the generated `pathdef` function resolvable, call it, and pass its returned character row to `path`." },
    BuiltinDocumentationFaq { question: "Can I keep several path profiles?", answer: "Yes. Save each profile to a different file, then resolve the desired generated function and pass its result to `path`." },
    BuiltinDocumentationFaq { question: "Does the saved path include the current working folder?", answer: "It mirrors `path`, which stores the ordered search-path entries separately from the implicit current-folder lookup." },
];

const RELATED: &[&str] = &[
    "path", "addpath", "rmpath", "genpath", "cd", "copyfile", "delete", "dir", "exist", "fullfile",
    "getenv", "ls", "mkdir", "movefile", "pwd", "rmdir", "run", "setenv", "tempdir", "tempname",
    "which",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "run", target: BuiltinDocumentationLinkTarget::Builtin("run") },
    BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Builtin("which") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/savepath/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("savepath"),
    slug: Some("savepath"),
    summary: "Write the current search path to a pathdef source file.",
    description: "`savepath` persists the active ordered search path as a MATLAB-syntax `pathdef` function and reports whether the write succeeded.",
    keywords: &["savepath", "pathdef", "search path", "persist path"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Pathdef persistence runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/savepath/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Admission, targets, serialization, persistence, outputs, and compatibility", location: "builtins::io::repl_fs::savepath::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Isolated filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
