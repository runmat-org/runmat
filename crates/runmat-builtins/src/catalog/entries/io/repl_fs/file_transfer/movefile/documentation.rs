use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Move and rename files",
        paragraphs: &[
            "`movefile(source, destination)` renames one file or directory or moves it to another directory. A source containing wildcards moves every match into an existing destination directory.",
            "Pass `'f'` as the third argument to replace an existing destination. Without that flag, an existing target is preserved and the operation reports failure.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Results and execution",
        paragraphs: &[
            "`status` is a double scalar: 1 after success and 0 after an operational failure. `msg` and `msgID` are empty character rows after success and describe a failure otherwise. Invalid argument types and malformed flags raise an error before filesystem access.",
            "Paths accept character rows and scalar strings. Relative paths use the current folder and a leading `~` expands through the active environment. The operation runs through the host or browser filesystem service; provider-resident inputs gather before parsing, and no accelerator kernel is used.",
            "The active filesystem provider defines cross-volume behavior and metadata preservation. If its rename operation cannot move between two storage volumes, movefile returns status 0 with the provider error and leaves recovery to the caller.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "rename",
        title: "Rename a file",
        program: "root = tempname();\nmkdir(root);\nsource = fullfile(root, 'results.txt');\ntarget = fullfile(root, 'archive.txt');\nfid = fopen(source, 'w'); fclose(fid);\nstatus = movefile(source, target);\nsourceAfter = exist(source, 'file');\ntargetAfter = exist(target, 'file');\nrmdir(root, 's');",
        display_output: Some("status is double 1, the source is absent, and the target exists"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(status, 'double') && status == 1); assert(sourceAfter == 0 && targetAfter ~= 0);" },
    },
    BuiltinExample {
        id: "existing-directory",
        title: "Move a file into a directory",
        program: "root = tempname();\nmkdir(root);\nmkdir(root, 'reports');\nsource = fullfile(root, 'summary.txt');\nfid = fopen(source, 'w'); fclose(fid);\nstatus = movefile(source, fullfile(root, 'reports'));\nmoved = exist(fullfile(root, 'reports', 'summary.txt'), 'file');\nrmdir(root, 's');",
        display_output: Some("the source name is retained inside the destination directory"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(moved ~= 0);" },
    },
    BuiltinExample {
        id: "force",
        title: "Replace an existing file",
        program: "root = tempname();\nmkdir(root);\nsource = fullfile(root, 'draft.txt');\ntarget = fullfile(root, 'final.txt');\nfid = fopen(source, 'w'); fclose(fid);\nfid = fopen(target, 'w'); fclose(fid);\n[status, msg, msgID] = movefile(source, target, 'f');\nrmdir(root, 's');",
        display_output: Some("status is 1 and both diagnostic character rows are empty"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(ischar(msg) && isempty(msg)); assert(ischar(msgID) && isempty(msgID));" },
    },
    BuiltinExample {
        id: "wildcard",
        title: "Move every wildcard match",
        program: "root = tempname();\nmkdir(root);\nmkdir(root, 'data');\nfid = fopen(fullfile(root, 'a.log'), 'w'); fclose(fid);\nfid = fopen(fullfile(root, 'b.log'), 'w'); fclose(fid);\nstatus = movefile(fullfile(root, '*.log'), fullfile(root, 'data'));\nfirst = exist(fullfile(root, 'data', 'a.log'), 'file');\nsecond = exist(fullfile(root, 'data', 'b.log'), 'file');\nrmdir(root, 's');",
        display_output: Some("both matching files are moved"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(first ~= 0 && second ~= 0);" },
    },
    BuiltinExample {
        id: "missing-source",
        title: "Inspect a missing-source result",
        program: "root = tempname();\nmkdir(root);\n[status, msg, msgID] = movefile(fullfile(root, 'missing.txt'), fullfile(root, 'dest.txt'));\nrmdir(root);",
        display_output: Some("status is 0 and the diagnostic rows identify the failure"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(ischar(msg) && ~isempty(msg)); assert(ischar(msgID) && ~isempty(msgID));" },
    },
    BuiltinExample {
        id: "same-path",
        title: "Leave a file in place when both paths match",
        program: "root = tempname();\nmkdir(root);\nsource = fullfile(root, 'stable.txt');\nfid = fopen(source, 'w'); fclose(fid);\nstatus = movefile(source, source);\nstillPresent = exist(source, 'file');\nrmdir(root, 's');",
        display_output: Some("status is 1 and the file remains at the same path"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(stillPresent ~= 0);" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do I replace an existing target?", answer: "Pass `'f'` as the third argument. Without it, movefile leaves the existing target unchanged and returns status 0." },
    BuiltinDocumentationFaq { question: "Can movefile move several files?", answer: "Yes. Use a wildcard source and an existing destination directory. Every match is planned before the first move begins." },
    BuiltinDocumentationFaq { question: "Can movefile cross storage volumes?", answer: "Only when the active filesystem provider supports that rename. An unsupported cross-volume move returns status 0 and provider diagnostics." },
    BuiltinDocumentationFaq { question: "Can movefile run on a GPU?", answer: "No. Filesystem access is a host-service operation. Resident arguments gather before validation and execution." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "savepath", target: BuiltinDocumentationLinkTarget::Builtin("savepath") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/movefile/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("movefile"),
    slug: Some("movefile"),
    summary: "Move or rename files, directories, or wildcard matches.",
    description: "`movefile` changes filesystem paths and returns status with optional diagnostics.",
    keywords: &["movefile", "rename", "move file", "filesystem", "status", "message", "messageid", "wildcard", "force", "overwrite"],
    related: &["copyfile", "mkdir", "rmdir", "dir", "pwd", "addpath", "cd", "delete", "exist", "fullfile", "genpath", "getenv", "ls", "path", "rmpath", "savepath", "setenv", "tempdir", "tempname"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Filesystem move runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/movefile/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Move, wildcard, overwrite, output, and filesystem-provider behavior", location: "builtins::io::repl_fs::file_transfer::movefile::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
