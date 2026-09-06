use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Copy files and directories",
        paragraphs: &[
            "`copyfile(source, destination)` copies one file or directory. A directory source includes its descendants. A source containing wildcards copies every match into an existing destination directory.",
            "Pass `'f'` as the third argument to replace an existing destination. Without that flag, an existing target is preserved and the operation reports failure.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Results and execution",
        paragraphs: &[
            "`status` is a double scalar: 1 after success and 0 after an operational failure. `msg` and `msgID` are empty character rows after success and describe a failure otherwise. Invalid argument types and malformed flags raise an error before filesystem access.",
            "Paths accept character rows and scalar strings. Relative paths use the current folder and a leading `~` expands through the active environment. The operation runs through the host or browser filesystem service; provider-resident inputs gather before parsing, and no accelerator kernel is used.",
            "File contents and directory trees are copied. Read-only state is preserved where the filesystem exposes it; other timestamps, permissions, links, and platform metadata remain subject to the active filesystem provider.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "rename-copy",
        title: "Copy a file to a new name",
        program: "root = tempname();\nmkdir(root);\nsource = fullfile(root, 'report.txt');\ntarget = fullfile(root, 'report-copy.txt');\nfid = fopen(source, 'w'); fclose(fid);\nstatus = copyfile(source, target);\ncopied = exist(target, 'file');\nrmdir(root, 's');",
        display_output: Some("status is double 1 and the destination file exists"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(status, 'double') && status == 1); assert(copied ~= 0);" },
    },
    BuiltinExample {
        id: "existing-directory",
        title: "Copy a file into a directory",
        program: "root = tempname();\nmkdir(root);\nmkdir(root, 'archive');\nsource = fullfile(root, 'summary.txt');\nfid = fopen(source, 'w'); fclose(fid);\nstatus = copyfile(source, fullfile(root, 'archive'));\ncopied = exist(fullfile(root, 'archive', 'summary.txt'), 'file');\nrmdir(root, 's');",
        display_output: Some("the source name is retained inside the destination directory"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(copied ~= 0);" },
    },
    BuiltinExample {
        id: "force",
        title: "Replace an existing file",
        program: "root = tempname();\nmkdir(root);\nsource = fullfile(root, 'draft.txt');\ntarget = fullfile(root, 'final.txt');\nfid = fopen(source, 'w'); fclose(fid);\nfid = fopen(target, 'w'); fclose(fid);\n[status, msg, msgID] = copyfile(source, target, 'f');\nrmdir(root, 's');",
        display_output: Some("status is 1 and both diagnostic character rows are empty"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(ischar(msg) && isempty(msg)); assert(ischar(msgID) && isempty(msgID));" },
    },
    BuiltinExample {
        id: "directory-tree",
        title: "Copy a directory tree",
        program: "root = tempname();\nmkdir(root, fullfile('data', 'raw'));\nsource = fullfile(root, 'data');\nfid = fopen(fullfile(source, 'raw', 'sample.dat'), 'w'); fclose(fid);\ntarget = fullfile(root, 'data-copy');\nstatus = copyfile(source, target);\ncopied = exist(fullfile(target, 'raw', 'sample.dat'), 'file');\nrmdir(root, 's');",
        display_output: Some("the nested file is copied with the directory tree"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(copied ~= 0);" },
    },
    BuiltinExample {
        id: "wildcard",
        title: "Copy every wildcard match",
        program: "root = tempname();\nmkdir(root);\nmkdir(root, 'logs');\nfid = fopen(fullfile(root, 'a.log'), 'w'); fclose(fid);\nfid = fopen(fullfile(root, 'b.log'), 'w'); fclose(fid);\nstatus = copyfile(fullfile(root, '*.log'), fullfile(root, 'logs'));\nfirst = exist(fullfile(root, 'logs', 'a.log'), 'file');\nsecond = exist(fullfile(root, 'logs', 'b.log'), 'file');\nrmdir(root, 's');",
        display_output: Some("both matching files are copied"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 1); assert(first ~= 0 && second ~= 0);" },
    },
    BuiltinExample {
        id: "missing-source",
        title: "Inspect a missing-source result",
        program: "root = tempname();\nmkdir(root);\n[status, msg, msgID] = copyfile(fullfile(root, 'missing.txt'), fullfile(root, 'dest.txt'));\nrmdir(root);",
        display_output: Some("status is 0 and the diagnostic rows identify the failure"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(ischar(msg) && ~isempty(msg)); assert(ischar(msgID) && ~isempty(msgID));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do I replace an existing target?", answer: "Pass `'f'` as the third argument. Without it, copyfile leaves the existing target unchanged and returns status 0." },
    BuiltinDocumentationFaq { question: "Can copyfile create destination parents?", answer: "No. Create missing parent directories before the call. A directory named by destination may be created when its parent already exists." },
    BuiltinDocumentationFaq { question: "What happens when source and destination identify the same path?", answer: "The operation returns status 0, leaves the path unchanged, and supplies a SourceEqualsDestination diagnostic identifier." },
    BuiltinDocumentationFaq { question: "Can copyfile run on a GPU?", answer: "No. Filesystem access is a host-service operation. Resident arguments gather before validation and execution." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
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
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/copyfile/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("copyfile"),
    slug: Some("copyfile"),
    summary: "Copy files, directory trees, or wildcard matches.",
    description: "`copyfile` duplicates filesystem content and returns status with optional diagnostics.",
    keywords: &["copyfile", "copy file", "copy folder", "filesystem", "status", "message", "messageid", "wildcard", "force", "overwrite"],
    related: &["movefile", "mkdir", "rmdir", "dir", "pwd", "addpath", "cd", "delete", "exist", "fullfile", "genpath", "getenv", "ls", "path", "rmpath", "savepath", "setenv", "tempdir", "tempname"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Filesystem copy runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/file_transfer/copyfile/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Copy, wildcard, overwrite, output, and filesystem-provider behavior", location: "builtins::io::repl_fs::file_transfer::copyfile::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
