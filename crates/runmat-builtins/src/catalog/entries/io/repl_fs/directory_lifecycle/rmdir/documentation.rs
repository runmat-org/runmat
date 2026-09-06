use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Remove directories",
        paragraphs: &[
            "`rmdir(folderName)` removes an empty directory. Add `'s'` to remove its descendants recursively. With outputs, the logical `status` reports success and `msg` and `msgID` are empty character rows after a successful removal.",
            "Operational failures return logical false and diagnostics when outputs are requested. Without outputs, they raise an error so a failed removal cannot pass silently.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Symbolic links and execution",
        paragraphs: &[
            "`ResolveSymbolicLinks=false` removes a symbolic link itself and leaves its target in place. `ResolveSymbolicLinks=true` removes the resolved target and leaves the link. The option accepts logical scalars and exact numeric zero or one; it may follow the optional `'s'` flag.",
            "The `'s'` flag and `ResolveSymbolicLinks` option name are case-insensitive. Relative paths use the current folder, a leading `~` expands through the active environment, and native Windows paths may use drive-letter or UNC syntax.",
            "Removal runs through the host or browser filesystem service and is never sent to an acceleration provider. Provider-resident arguments gather before parsing and filesystem access.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "empty",
        title: "Remove an empty directory",
        program: "folder = tempname();\nmkdir(folder);\nstatus = rmdir(folder);\nexistsAfter = exist(folder, 'dir');",
        display_output: Some("status is logical true and the folder is absent"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(existsAfter == 0);",
        },
    },
    BuiltinExample {
        id: "recursive",
        title: "Remove a directory hierarchy",
        program: "folder = tempname();\nmkdir(folder, fullfile('data', 'archive'));\nstatus = rmdir(folder, 's');\nexistsAfter = exist(folder, 'dir');",
        display_output: Some("status is logical true and the hierarchy is absent"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(existsAfter == 0);",
        },
    },
    BuiltinExample {
        id: "missing",
        title: "Capture an operational failure",
        program: "folder = tempname();\n[status, msg, msgID] = rmdir(folder);",
        display_output: Some("status is logical false and diagnostics identify the missing folder"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && ~status); assert(~isempty(msg)); assert(~isempty(msgID));",
        },
    },
    BuiltinExample {
        id: "nonempty",
        title: "Preserve a nonempty directory without recursion",
        program: "folder = tempname();\nmkdir(folder);\nfid = fopen(fullfile(folder, 'data.bin'), 'w');\nfclose(fid);\n[status, msg, msgID] = rmdir(folder);\nstillPresent = exist(folder, 'dir');\nrmdir(folder, 's');",
        display_output: Some("status is logical false and the nonempty directory remains"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && ~status); assert(~isempty(msg)); assert(~isempty(msgID)); assert(stillPresent ~= 0);",
        },
    },
    BuiltinExample {
        id: "uppercase-recursive",
        title: "Use the case-insensitive recursive flag",
        program: "folder = tempname();\nmkdir(folder, 'nested');\nstatus = rmdir(folder, 'S');\nexistsAfter = exist(folder, 'dir');",
        display_output: Some("uppercase S recursively removes the hierarchy"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(existsAfter == 0);",
        },
    },
    BuiltinExample {
        id: "symbolic-link-option-default",
        title: "Supply the symbolic-link policy explicitly",
        program: "folder = tempname();\nmkdir(folder);\nstatus = rmdir(folder, ResolveSymbolicLinks=false);\nexistsAfter = exist(folder, 'dir');",
        display_output: Some("the explicit default policy removes the directory"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(existsAfter == 0);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I remove a nonempty directory?",
        answer: "Pass `'s'` as the second argument. Without it, `rmdir` leaves the directory unchanged and reports that it is not empty.",
    },
    BuiltinDocumentationFaq {
        question: "Does rmdir follow symbolic links?",
        answer: "Not by default. Set `ResolveSymbolicLinks=true` when the resolved target, rather than the link entry, should be removed.",
    },
    BuiltinDocumentationFaq {
        question: "Does rmdir throw on a filesystem failure?",
        answer: "A no-output call raises an error. Capture status, message, and message ID to handle the failure programmatically.",
    },
    BuiltinDocumentationFaq {
        question: "Is the recursive flag case-sensitive?",
        answer: "No. Character-row and scalar-string forms of `'s'` and `'S'` select recursive removal.",
    },
    BuiltinDocumentationFaq {
        question: "Why are message and message ID character arrays?",
        answer: "The compatibility interface returns character rows, including 1-by-0 empty rows after success. Convert them explicitly when a string value is needed.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "mkdir",
        target: BuiltinDocumentationLinkTarget::Builtin("mkdir"),
    },
    BuiltinDocumentationLink {
        label: "delete",
        target: BuiltinDocumentationLinkTarget::Builtin("delete"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/rmdir/mod.rs"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rmdir"),
    slug: Some("rmdir"),
    summary: "Remove directories with optional recursion and symbolic-link resolution.",
    description: "`rmdir` removes a directory through the active filesystem service and provides logical status with optional diagnostics.",
    keywords: &[
        "rmdir",
        "remove directory",
        "delete folder",
        "filesystem",
        "status",
        "message",
        "messageid",
        "recursive",
        "symbolic link",
    ],
    related: &["mkdir", "delete", "fullfile", "pwd", "exist"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Directory removal runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/rmdir/mod.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Directory, recursive, symbolic-link, output, and virtual-filesystem behavior",
                location: "builtins::io::repl_fs::directory_lifecycle::rmdir::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Executable filesystem examples",
                location: "scripts/runtime/verify-builtin-examples.mjs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
